"""Connect the experiment checkout to existing dataset and persistent results."""
import argparse
from pathlib import Path


def link_storage(workdir, data_root, results_root):
    workdir = Path(workdir).resolve()
    data_root = Path(data_root).resolve()
    results_root = Path(results_root).resolve()
    required = ['Own/X_4_raw.npy', 'Own/y_4_raw.npy',
                'Yuan/FLAT/X_3_raw.npy', 'Yuan/FLAT/y_3_raw.npy']
    missing = [str(data_root / name) for name in required
               if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing preprocessed dataset arrays:\n' + '\n'.join(missing))

    links = [(workdir / 'data', data_root), (workdir / 'results', results_root)]
    # Check both destinations before making changes. Keep existing data intact.
    for link, target in links:
        if link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f'Refusing to replace {link}; expected a link to {target}')
    if results_root.exists() and not results_root.is_dir():
        raise NotADirectoryError(str(results_root))

    results_root.mkdir(parents=True, exist_ok=True)
    for link, target in links:
        if link.resolve() != target:
            link.symlink_to(target, target_is_directory=True)
    print(f'Dataset: {data_root}\nResults: {results_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--results-root', required=True, type=Path)
    args = parser.parse_args()
    link_storage(Path(__file__).resolve().parents[1], args.data_root, args.results_root)
