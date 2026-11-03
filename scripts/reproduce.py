"""Run the paper's experiment groups; keep every attempt and resume finished runs."""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def groups(experiment):
    if experiment == 'model-comparison':
        for dataset, model in itertools.product(['hardness', 'yuan'], ['OFNN', 'DinoNN', 'ConvLSTM']):
            yield dataset, model, 'YOTO', .2, 2, 50
    elif experiment == 'dropout':
        for dropout, index in itertools.product(range(1, 11), range(9)):
            yield 'yuan', 'DinoNN', 'variance', dropout / 20, index, 1
    else:
        for dataset, model, classes, repeats in [('hardness', 'ConvLSTM', 5, 10), ('yuan', 'DinoNN', 9, 5)]:
            for strategy, index in itertools.product(['variance', 'entropy', 'random'], range(classes)):
                yield dataset, model, strategy, .2, index, repeats


def values(record, key):
    return [row[key] for row in record['history'] if key in row]


def valid(record, experiment):
    epochs = 100 if record['dataset'] == 'hardness' else 50
    if experiment == 'model-comparison':
        return all(len(values(record, k)) == epochs for k in ['train_accuracy', 'val_accuracy'])
    return record['converged'] and all(len(values(record, k)) == n for k, n in [
        ('acc_comp', 6), ('abs_diff_comp', 6), ('avg_entropy', 5), ('avg_variance', 5)])


def prepare(own, yuan):
    sys.path.insert(0, str(ROOT / 'hardness_classification'))
    from preprocessing import DatasetManager, DatasetManagerYuan
    managers = [DatasetManager([own], ['hardness'+str(i) for i in range(1, 6)], seq_length=4),
                DatasetManagerYuan(yuan, objects=['FLAT'],
                                   classes=[8, 17, 27, 46, 62, 73, 81, 87, 95], seq_length=3)]
    for manager in managers:
        manager.storage = str(ROOT / 'data')
        Path(manager.storage).mkdir(exist_ok=True)
        manager.get_raw()


def run(experiment, dry_run=False):
    mode = os.environ.get('WANDB_MODE', 'online')
    if not dry_run and mode == 'online':
        for key in ['WANDB_ENTITY', 'WANDB_API_KEY']:
            if not os.environ.get(key):
                raise ValueError(f'Set {key} for online tracking; see REPRODUCE.md')
    output = ROOT / 'results' / experiment
    output.mkdir(parents=True, exist_ok=True)
    # One campaign at a time also protects the shared dataset links and W&B files.
    with (ROOT / 'results/.training.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not dry_run:
            cache = ROOT / 'hardness_classification/processed_data'
            cache.mkdir(exist_ok=True)
            for dataset, suffix in [('Own', '4'), ('Yuan/FLAT', '3')]:
                for prefix in ['X_', 'y_']:
                    path = ROOT / 'data' / dataset / (prefix+suffix+'_raw.npy')
                    if not path.is_file():
                        raise FileNotFoundError(f'{path}; see REPRODUCE.md (data volume setup)')
            for dataset in ['Own', 'Yuan']:
                link, target = cache / dataset, ROOT / 'data' / dataset
                if link.is_symlink() and link.resolve() == target.resolve():
                    continue
                link.symlink_to(target, target_is_directory=True)
        for dataset, model, strategy, dropout, index, target in groups(experiment):
            folder = output / f'{dataset}_{model}_{strategy}_{dropout:.2f}_{index}'
            folder.mkdir(exist_ok=True)
            records = [json.loads(p.read_text()) for p in folder.glob('*/result.json')]
            complete = sum(valid(r, experiment) for r in records)
            for attempt in range(1, target * 5 + 1):
                if complete >= target:
                    break
                directory = folder / f'{attempt:03}'
                if directory.exists():
                    continue  # Retain failed/interrupted attempts as well.
                command = [sys.executable, '-u', 'leave_one_out.py', '--dataset', dataset,
                           '--model', model, '--strat', strategy, '--dropout', str(dropout),
                           '--comp_index', str(index), '--prefix', experiment, '--verbose', '2']
                if dry_run:
                    print(target, 'runs:', ' '.join(command))
                    break
                directory.mkdir()
                metadata = dict(experiment=experiment, dataset=dataset, model=model,
                                strategy=strategy, dropout=dropout, index=index, attempt=attempt)
                env = dict(os.environ, WANDB_MODE=mode, WANDB_DIR=str(directory),
                           TACTILE_RUN_CONFIG=json.dumps(metadata),
                           TACTILE_METRICS=str(directory / 'metrics.json'), TACTILE_SAVE_CHECKPOINT='0')
                with (directory / 'train.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT / 'hardness_classification', env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                record = json.loads((directory / 'metrics.json').read_text())
                record.update(dataset=dataset, model=model, strategy=strategy, dropout=dropout,
                              index=index, attempt=attempt)
                (directory / 'result.json').write_text(json.dumps(record))
                if experiment == 'model-comparison' and not valid(record, experiment):
                    raise RuntimeError(f'Incomplete baseline history: {directory}')
                complete += valid(record, experiment)
            if not dry_run and complete < target:
                raise RuntimeError(f'Attempt limit reached: {folder} ({complete}/{target})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['prepare', 'model-comparison', 'dropout', 'active-sampling'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--own', help='Own raw directory containing meta/, frames/, forces/')
    parser.add_argument('--yuan', help='Yuan Data1 directory')
    args = parser.parse_args()
    if args.experiment == 'prepare':
        if not args.own or not args.yuan:
            parser.error('prepare needs --own and --yuan')
        prepare(args.own, args.yuan)
    else:
        run(args.experiment, args.dry_run)
