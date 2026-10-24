"""CPU-only tests for quota accounting and persisted completion evidence."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('campaign',ROOT/'scripts/replenish_convlstm.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


class ReplenishmentTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((ROOT/'experiments/convlstm_replenishment.json').read_text())

    def test_audited_quotas(self):
        totals={s:sum(g['target_successes']-g['original_successes'] for g in self.plan['groups'] if g['strategy']==s)
                for s in ['variance','entropy','random']}
        self.assertEqual(totals,{'variance':9,'entropy':12,'random':7})
        self.assertEqual(sum(g['original_successes'] for g in self.plan['groups']),122)

    def test_does_not_schedule_a_pending_or_completed_group(self):
        self.assertEqual(campaign.choose_group(self.plan,[],{0}),(1,1))
        records=[{'group':0,'status':'success'}]
        self.assertEqual(campaign.choose_group(self.plan,records,set()),(1,1))
        records=[{'group':0,'status':'nonconverged'}]
        self.assertEqual(campaign.choose_group(self.plan,records,set()),(0,2))

    def test_completion_requires_all_groups_not_total_success_count(self):
        records=[]
        for i,g in enumerate(self.plan['groups']):
            records.extend({'group':i,'status':'success'} for _ in range(10-g['original_successes']))
        self.assertTrue(campaign.completed_all(self.plan,records))
        self.assertIsNone(campaign.choose_group(self.plan,records,set()))
        records[0]['status']='nonconverged'
        self.assertFalse(campaign.completed_all(self.plan,records))

    def test_incomplete_or_failed_trajectories_never_count(self):
        good={'converged':True,'completed_updates':5,'metric_counts':dict(campaign.SUCCESS_COUNTS),'wandb_finished':True}
        self.assertTrue(campaign.validate_result(good,0))
        bad={**good,'metric_counts':{**campaign.SUCCESS_COUNTS,'acc_comp':5}}
        with self.assertRaises(RuntimeError):campaign.validate_result(bad,0)
        with self.assertRaises(RuntimeError):campaign.validate_result(good,1)
        with self.assertRaises(RuntimeError):campaign.validate_result({**good,'wandb_finished':False},0)
        failed={'converged':False,'completed_updates':0,'metric_counts':dict(campaign.FAILURE_COUNTS),'wandb_finished':True}
        self.assertFalse(campaign.validate_result(failed,0))

    def test_result_is_written_only_after_successful_wandb_finish(self):
        tree=ast.parse((ROOT/'hardness_classification/adaptive_sampling.py').read_text())
        adaptive=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='adaptive_sampling')
        helper=next(n for n in adaptive.body if isinstance(n,ast.FunctionDef) and n.name=='finish_run')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json'
            run=SimpleNamespace(id='test',summary={},finish=lambda:None)
            scope={'run':run,'Path':Path,'json':json,'os':os,'metric_counts':dict(campaign.SUCCESS_COUNTS)}
            exec(compile(ast.Module(body=[helper],type_ignores=[]),'<finish_run>','exec'),scope)
            with patch.dict(os.environ,{'TACTILE_RESULT_PATH':str(path)}):
                scope['finish_run'](True,5)
                result=json.loads(path.read_text())
                self.assertTrue(campaign.validate_result(result,0))
                path.unlink()
                def fail():raise RuntimeError('network upload failed')
                run.finish=fail
                with self.assertRaises(RuntimeError):scope['finish_run'](True,5)
                self.assertFalse(path.exists())

    def test_controller_retries_only_nonconvergence_and_resumes_completed_quota(self):
        plan={**self.plan,'groups':[{**self.plan['groups'][0],'original_successes':9}]}
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            outcomes=iter([False,True])
            calls=[]
            def launch(command, **kwargs):
                converged=next(outcomes)
                calls.append(command)
                run_id=kwargs['env']['WANDB_RUN_ID']
                result={'run_id':run_id,'converged':converged,'completed_updates':5 if converged else 0,
                        'metric_counts':dict(campaign.SUCCESS_COUNTS if converged else campaign.FAILURE_COUNTS),
                        'wandb_finished':True}
                Path(kwargs['env']['TACTILE_RESULT_PATH']).write_text(json.dumps(result))
                return SimpleNamespace(pid=10000+len(calls),poll=lambda:0)
            env={'WANDB_MODE':'online','WANDB_API_KEY':'test-key','WANDB_PROJECT':'test-project'}
            with patch.dict(os.environ,env), patch.object(campaign,'digest',return_value='test'), \
                 patch.object(campaign,'verify_wandb') as verify, \
                 patch.object(campaign.subprocess,'Popen',side_effect=launch), \
                 patch.object(campaign.time,'sleep'):
                campaign.run_campaign(plan,output,1,3)
                self.assertEqual(len(calls),2)
                self.assertEqual(verify.call_count,2)
                records=[json.loads(p.read_text()) for p in sorted(output.glob('attempt-*/attempt.json'))]
                self.assertEqual([r['status'] for r in records],['nonconverged','success'])
                self.assertNotEqual(records[0]['seed'],records[1]['seed'])
                self.assertTrue(json.loads((output/'progress.json').read_text())['complete'])
                campaign.run_campaign(plan,output,1,3)
                self.assertEqual(len(calls),2)  # no duplicate work on resume

    def test_two_workers_have_private_paths_and_do_not_overshoot(self):
        plan={**self.plan,'groups':[{**g,'original_successes':9} for g in self.plan['groups'][:2]]}
        calls=[]
        with tempfile.TemporaryDirectory() as temporary:
            def launch(command, **kwargs):
                env=kwargs['env'];calls.append(kwargs)
                result={'run_id':env['WANDB_RUN_ID'],'converged':True,'completed_updates':5,
                        'metric_counts':dict(campaign.SUCCESS_COUNTS),'wandb_finished':True}
                Path(env['TACTILE_RESULT_PATH']).write_text(json.dumps(result))
                polls=iter([None,0])
                return SimpleNamespace(pid=12000+len(calls),poll=lambda:next(polls,0))
            env={'WANDB_MODE':'online','WANDB_API_KEY':'test-key','WANDB_PROJECT':'test-project'}
            with patch.dict(os.environ,env), patch.object(campaign,'digest',return_value='test'), \
                 patch.object(campaign,'verify_wandb'), \
                 patch.object(campaign.subprocess,'Popen',side_effect=launch), \
                 patch.object(campaign.time,'sleep'):
                campaign.run_campaign(plan,Path(temporary),2,4)
            self.assertEqual(len(calls),2)
            self.assertNotEqual(calls[0]['cwd'],calls[1]['cwd'])
            for key in ['WANDB_RUN_ID','PYTHONHASHSEED','TMPDIR','CUDA_CACHE_PATH','KERAS_HOME',
                        'MPLCONFIGDIR','WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR']:
                self.assertNotEqual(calls[0]['env'][key],calls[1]['env'][key],key)
            for call in calls:
                self.assertEqual(call['env']['TACTILE_READONLY_DATA'],'1')
                self.assertEqual(call['env']['PYTHONDONTWRITEBYTECODE'],'1')
            self.assertTrue(json.loads((Path(temporary)/'progress.json').read_text())['complete'])

    def test_sample_selection_is_repeatable_with_recorded_seed(self):
        import random
        import numpy as np
        tree=ast.parse((ROOT/'hardness_classification/preprocessing.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DatasetManager')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_raw_with_classes')
        scope={'random':random,'np':np,'os':os}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<sampler>','exec'),scope)
        dummy=SimpleNamespace(get_raw=lambda:(np.arange(100),np.array(['hardness1']*100)))
        def sequence(seed):
            random.seed(seed)
            return [scope['get_raw_with_classes'](dummy,['hardness1'],5,shuffle=True)[0].tolist() for _ in range(3)]
        with patch.dict(os.environ,{'TACTILE_RUN_SEED':'123'}):
            self.assertEqual(sequence(123),sequence(123))
            self.assertNotEqual(sequence(123),sequence(124))
            draws=sequence(123)
            self.assertNotEqual(draws[0],draws[1])

if __name__=='__main__':unittest.main()
