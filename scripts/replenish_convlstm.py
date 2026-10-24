"""Run/resume audited per-class ConvLSTM completion quotas, with all attempts logged.

Uses only stdlib until online verification is needed; training runs in separate
processes using the locked interpreter and isolated working directories.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Keep the invoked repository path even when SkyPilot symlinks this script.
ROOT = Path(__file__).absolute().parents[1]
SUCCESS_COUNTS = {'train_accuracy':350,'acc_comp':6,'abs_diff_comp':6,'avg_entropy':5,'avg_variance':5}
FAILURE_COUNTS = {'train_accuracy':100,'acc_comp':1,'abs_diff_comp':1,'avg_entropy':0,'avg_variance':0}


def save_json(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,default=str)+'\n')
    temporary.replace(path)


def digest(path):
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):hasher.update(chunk)
    return hasher.hexdigest()


def validate_result(result, exit_code):
    if exit_code != 0:raise RuntimeError('Training process failed; stopping instead of retrying infrastructure errors')
    if result.get('converged') not in [True,False] or not isinstance(result.get('converged'),bool):
        raise RuntimeError('Missing convergence outcome')
    expected=SUCCESS_COUNTS if result['converged'] else FAILURE_COUNTS
    if result.get('metric_counts')!=expected:
        raise RuntimeError('Incomplete/unexpected metric histories: '+str(result.get('metric_counts')))
    if result.get('completed_updates')!=(5 if result['converged'] else 0) or result.get('wandb_finished') is not True:
        raise RuntimeError('Incomplete trajectory or W&B finish')
    return result['converged']


def verify_wandb(run_id, result, entity, project):
    import wandb
    expected=SUCCESS_COUNTS if result['converged'] else FAILURE_COUNTS
    last_error=None
    for attempt in range(6):
        try:
            run=wandb.Api(timeout=30).run(entity+'/'+project+'/'+run_id)
            if run.state!='finished':raise RuntimeError('W&B state is '+run.state)
            if run.summary.get('converged') is not result['converged']:
                raise RuntimeError('W&B convergence flag disagrees')
            if run.summary.get('completed_updates')!=result['completed_updates']:
                raise RuntimeError('W&B update count disagrees')
            counts={k:0 for k in expected}
            for row in run.scan_history(page_size=1000):
                for key in counts:
                    if key in row and row[key] is not None:
                        if not isinstance(row[key],(int,float)) or not math.isfinite(row[key]):
                            raise RuntimeError('Non-finite/non-numeric W&B metric: '+key)
                        counts[key]+=1
            if counts!=expected:raise RuntimeError('W&B metric counts disagree: '+str(counts))
            return
        except Exception as error:
            last_error=error
            if attempt<5:time.sleep(5)
    raise RuntimeError('Cannot verify synchronized W&B run '+run_id+': '+str(last_error))


def choose_group(plan, records, pending_groups):
    completed=Counter(r['group'] for r in records if r.get('status')=='success')
    attempts=Counter(r['group'] for r in records)
    for index,group in enumerate(plan['groups']):
        missing=group['target_successes']-group['original_successes']-completed[index]
        if missing>0 and index not in pending_groups:
            return index,attempts[index]+1
    return None


def completed_all(plan, records):
    counts=Counter(r['group'] for r in records if r.get('status')=='success')
    return all(g['original_successes']+counts[i]==g['target_successes'] for i,g in enumerate(plan['groups']))


def run_campaign(plan, output, workers, max_attempts):
    if workers not in [1,2]:raise ValueError('Use 1 or 2 workers; benchmark before increasing further')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=output/'plan.json'
    if frozen.exists() and json.loads(frozen.read_text())!=plan:
        raise RuntimeError('Campaign plan changed; refusing to mix plans')
    save_json(frozen,plan)
    env=os.environ.copy()
    if env.get('WANDB_MODE')!='online' or not env.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B logging with an API key is required')
    entity=env.get('WANDB_ENTITY','patchouli_lab')
    project=env.get('WANDB_PROJECT','tactile-hardness-replenishment')
    provenance={
        'campaign':plan['campaign'],'entity':entity,'project':project,'workers':workers,
        'image':plan['image'],'cohort':plan['cohort'],
        'source_sha256':{p.name:digest(p) for p in [ROOT/'hardness_classification'/n for n in
          ['leave_one_out.py','adaptive_sampling.py','models.py','preprocessing.py']]},
        'dataset_sha256':{n:digest(Path('/workspace/data/Own')/n) for n in ['X_4_raw.npy','y_4_raw.npy']},
        'locks_sha256':{n:digest(ROOT/'environment'/n) for n in ['conda-linux-64.lock.txt','requirements-linux-64.lock.txt']},
        'wandb_patch_sha256':digest(ROOT/'scripts/patch_wandb_0179.py'),
        'training_settings':plan['settings'],
        'pythonhashseed':'Per-attempt seed, set before process startup',
        'shared_dataset_policy':'read-only arrays; regeneration disabled',
        'worker_caches':'Isolated per-attempt temp, CUDA, Keras, Matplotlib and W&B cache/config/data directories',
        'seed_policy':'Python/NumPy/TensorFlow seeded; sampler entropy reseeding replaced by deterministic derived seeds only when --seed is supplied; bitwise GPU reproducibility not guaranteed',
    }
    provpath=output/'provenance.json'
    if provpath.exists():
        previous=json.loads(provpath.read_text())
        for key in ['entity','project','source_sha256','dataset_sha256','locks_sha256','wandb_patch_sha256']:
            if previous[key]!=provenance[key]:raise RuntimeError('Campaign provenance changed: '+key)
    else:save_json(provpath,provenance)
    records=[]
    for p in sorted(output.glob('attempt-*/attempt.json')):
        record=json.loads(p.read_text())
        if record['status']=='running':
            result_path=p.parent/'result.json'
            if not result_path.exists():
                raise RuntimeError('Interrupted attempt needs reconciliation before resume: '+str(p.parent))
            result=json.loads(result_path.read_text())
            # A persisted exit code is required; an interrupted controller must not
            # assume that run.finish implies the entire process exited cleanly.
            exit_path=p.parent/'process-exit.json'
            if not exit_path.exists():raise RuntimeError('Interrupted attempt needs process-exit reconciliation: '+str(p.parent))
            valid=validate_result(result,json.loads(exit_path.read_text())['exit_code'])
            verify_wandb(record['run_id'],result,entity,project)
            record.update(status='success' if valid else 'nonconverged',result=result)
            save_json(p,record)
        if record['status']=='error':raise RuntimeError('Unresolved failed attempt: '+str(p.parent))
        records.append(record)
    active={}
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    old={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    def update_progress():
        completed=Counter(r['group'] for r in records if r['status']=='success')
        progress=[]
        for i,g in enumerate(plan['groups']):
            progress.append({'strategy':g['strategy'],'class_index':g['class_index'],
                'original_successes':g['original_successes'],'new_successes':completed[i],
                'remaining':g['target_successes']-g['original_successes']-completed[i]})
        save_json(output/'progress.json',{'groups':progress,'total_attempts':len(records),
            'complete':completed_all(plan,records),'updated_unix':time.time()})
        return sum(g['remaining'] for g in progress)
    try:
        while True:
            if stopping:raise RuntimeError('Controller interrupted; no further attempts will start')
            while len(active)<workers and len(records)<max_attempts:
                selected=choose_group(plan,records,{r['group'] for _,r,_,_ in active.values()})
                if selected is None:break
                index,attempt=selected;group=plan['groups'][index]
                seed=group['seed_start']+attempt
                run_id=hashlib.sha256((plan['campaign']+':'+str(index)+':'+str(attempt)).encode()).hexdigest()[:12]
                directory=output/('attempt-%02d-%03d'%(index,attempt));directory.mkdir(exist_ok=False)
                (directory/'processed_data').mkdir()
                (directory/'processed_data/Own').symlink_to('/workspace/data/Own',target_is_directory=True)
                for name in ['logs','model_dumps','AdaptiveModels']:(directory/name).mkdir()
                # Give every library a private writable cache/config/temp directory.
                # The dataset symlink is the only shared input; it is never regenerated.
                cache=directory/'cache';cache.mkdir()
                temporary=directory/'tmp';temporary.mkdir()
                run_env=env.copy();run_env.update(PYTHONHASHSEED=str(seed),WANDB_RUN_ID=run_id,
                    WANDB_RESUME='never',WANDB_DIR=str(directory),WANDB_TAGS='replenishment,tf210-reconstruction',
                    TACTILE_RESULT_PATH=str(directory/'result.json'),TACTILE_READONLY_DATA='1',
                    TMPDIR=str(temporary),XDG_CACHE_HOME=str(cache),PYTHONDONTWRITEBYTECODE='1',
                    MPLCONFIGDIR=str(cache/'matplotlib'),KERAS_HOME=str(cache/'keras'),
                    CUDA_CACHE_PATH=str(cache/'cuda'),WANDB_CACHE_DIR=str(cache/'wandb'),
                    WANDB_CONFIG_DIR=str(cache/'wandb-config'),WANDB_DATA_DIR=str(cache/'wandb-data'))
                metadata={**provenance,'seed':seed,'strategy':group['strategy'],'comparison_class':'hardness'+str(group['class_index']+1),
                    'attempt_in_group':attempt,'original_project':plan['original_project'],
                    'original_failed_run_ids':group['original_failed_run_ids'],'quota_target_per_group':10,
                    'original_successes_in_group':group['original_successes'],'campaign_id':plan['campaign']}
                run_env['TACTILE_RUN_CONFIG_JSON']=json.dumps(metadata)
                command=[sys.executable,'-u',str(ROOT/'hardness_classification/leave_one_out.py'),
                    '--model','ConvLSTM','--dataset','hardness','--strat',group['strategy'],
                    '--comp_index',str(group['class_index']),'--dropout','0.2','--seed',str(seed),
                    '--no-checkpoint','--prefix',plan['campaign'],'--verbose','2']
                record={'group':index,'attempt_in_group':attempt,'seed':seed,'run_id':run_id,
                    'status':'running','started_unix':time.time(),'command':command}
                save_json(directory/'attempt.json',record)
                logfile=(directory/'train.log').open('w')
                process=subprocess.Popen(command,cwd=directory,env=run_env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                records.append(record);active[process.pid]=(process,record,directory,logfile)
                print('Started',group['strategy'],'class',group['class_index']+1,'seed',seed,'W&B',run_id,flush=True)
            update_progress()
            if not active:
                if completed_all(plan,records):
                    print('COMPLETE: 150 successful original + replenishment trajectories.',flush=True)
                    return
                raise RuntimeError('Reached attempt cap before quotas were filled; inspect progress.json')
            for pid,(process,record,directory,logfile) in list(active.items()):
                code=process.poll()
                if code is None:continue
                logfile.close();active.pop(pid)
                save_json(directory/'process-exit.json',{'exit_code':code})
                try:
                    result=json.loads((directory/'result.json').read_text())
                    success=validate_result(result,code)
                    verify_wandb(record['run_id'],result,entity,project)
                except Exception as error:
                    record.update(status='error',error=str(error),finished_unix=time.time())
                    save_json(directory/'attempt.json',record);update_progress();raise
                record.update(status='success' if success else 'nonconverged',result=result,finished_unix=time.time())
                save_json(directory/'attempt.json',record)
                remaining=update_progress()
                print('Finished',record['run_id'],record['status'],'remaining successes',remaining,flush=True)
            time.sleep(2)
    finally:
        for process,record,directory,logfile in active.values():
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        for process,record,directory,logfile in active.values():
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            logfile.close()
        for sig,handler in old.items():signal.signal(sig,handler)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'experiments/convlstm_replenishment.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,choices=[1,2],default=1)
    parser.add_argument('--max-attempts',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert len(plan['groups'])==15 and sum(g['original_successes'] for g in plan['groups'])==122
    required=sum(g['target_successes']-g['original_successes'] for g in plan['groups'])
    assert required==28 and all(0<=g['original_successes']<=g['target_successes']==10 for g in plan['groups'])
    if args.dry_run:
        print(json.dumps({'campaign':plan['campaign'],'workers':args.workers,'additional_successes':required,
            'maximum_new_attempts':args.max_attempts or plan['maximum_new_attempts'],'groups':plan['groups']},indent=2));return
    output=args.output or Path('/workspace/runs')/plan['campaign']
    run_campaign(plan,output.resolve(),args.workers,args.max_attempts or plan['maximum_new_attempts'])

if __name__=='__main__':main()
