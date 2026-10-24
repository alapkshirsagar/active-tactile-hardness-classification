"""Compatibility fix for W&B 0.17.9 against servers without ServerSettings.

Only the optional schema probe changes. Missing/null types yield no supported
fields. Authentication, run creation, metric uploads, and errors remain enabled.
"""
import importlib.metadata
import importlib.util
from pathlib import Path

OLD = 'res.get("ServerSettingsType", {}).get("fields", [{}])'
NEW = '(res.get("ServerSettingsType") or {}).get("fields", [])'


def main():
    version = importlib.metadata.version('wandb')
    if version != '0.17.9':
        raise RuntimeError('This compatibility fix is specific to wandb==0.17.9; found '+version)
    spec = importlib.util.find_spec('wandb')
    path = Path(spec.origin).parent/'sdk/internal/internal_api.py'
    source = path.read_text()
    if NEW in source and OLD not in source:
        print('W&B optional ServerSettings null-handling fix already applied.')
        return
    if source.count(OLD) != 1:
        raise RuntimeError('Unexpected W&B source; refusing to patch '+str(path))
    updated = source.replace(OLD, NEW)
    compile(updated, str(path), 'exec')
    path.write_text(updated)
    print('Applied W&B 0.17.9 optional ServerSettings null-handling fix.')


if __name__ == '__main__':
    main()
