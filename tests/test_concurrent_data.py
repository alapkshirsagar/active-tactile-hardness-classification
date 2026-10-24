"""Exercise the real cached-data loader in two independent OS processes."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'hardness_classification/preprocessing.py'


def cached_loader():
    tree=ast.parse(SOURCE.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DatasetManager')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_raw')
    scope={'np':np,'os':os,'join':os.path.join,'exists':os.path.exists}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope['get_raw']


CHILD=r'''
import ast, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
source=Path(sys.argv[1]);other=Path(sys.argv[2])
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DatasetManager')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_raw')
scope={'np':np,'os':os,'join':os.path.join,'exists':os.path.exists}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
def forbidden_save(*args,**kwargs):raise AssertionError('Loader tried to regenerate shared arrays')
np.save=forbidden_save
manager=SimpleNamespace(storage='processed_data',id='Own',seq_length=4,paths=[])
Path('ready').touch()
deadline=time.monotonic()+20
while not other.exists():
    if time.monotonic()>deadline:raise RuntimeError('Other reader did not start')
    time.sleep(.01)
for _ in range(20):
    x,y=scope['get_raw'](manager)
    assert np.array_equal(x,np.arange(640).reshape(20,4,8))
    assert y.tolist()==['hardness1']*20
    x[:]=-1  # Each process owns its array; this must not alter the shared file.
    time.sleep(.005)
print('20 concurrent reads passed')
'''


class ConcurrentDataTests(unittest.TestCase):
    def test_two_processes_read_same_arrays_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shared=root/'data/Own';shared.mkdir(parents=True)
            np.save(shared/'X_4_raw.npy',np.arange(640).reshape(20,4,8))
            np.save(shared/'y_4_raw.npy',np.array(['hardness1']*20))
            for p in shared.iterdir():p.chmod(0o444)
            before={p.name:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns) for p in shared.iterdir()}
            workers=[root/'worker-1',root/'worker-2']
            for worker in workers:
                (worker/'processed_data').mkdir(parents=True)
                (worker/'processed_data/Own').symlink_to(shared,target_is_directory=True)
            env={**os.environ,'TACTILE_READONLY_DATA':'1','OPENBLAS_NUM_THREADS':'1'}
            processes=[]
            try:
                for i,worker in enumerate(workers):
                    processes.append(subprocess.Popen([sys.executable,'-c',CHILD,str(SOURCE),str(workers[1-i]/'ready')],
                        cwd=worker,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
                for process in processes:
                    stdout,stderr=process.communicate(timeout=30)
                    self.assertEqual(process.returncode,0,stderr)
                    self.assertIn('20 concurrent reads passed',stdout)
            finally:
                for process in processes:
                    if process.poll() is None:process.kill();process.wait()
            after={p.name:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns) for p in shared.iterdir()}
            self.assertEqual(before,after)

    def test_missing_cache_fails_before_attempting_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            manager=SimpleNamespace(storage=directory,id='Own',seq_length=4,paths=[])
            loader=cached_loader()
            with patch.dict(os.environ,{'TACTILE_READONLY_DATA':'1'}):
                with patch.object(np,'save',side_effect=AssertionError('Unexpected cache write')):
                    with self.assertRaisesRegex(FileNotFoundError,'refusing to regenerate'):
                        loader(manager)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_incomplete_pair_does_not_overwrite_existing_array(self):
        with tempfile.TemporaryDirectory() as directory:
            shared=Path(directory)/'Own';shared.mkdir()
            path=shared/'X_4_raw.npy';np.save(path,np.arange(4));original=path.read_bytes()
            manager=SimpleNamespace(storage=directory,id='Own',seq_length=4,paths=[])
            with patch.dict(os.environ,{'TACTILE_READONLY_DATA':'1'}):
                with self.assertRaises(FileNotFoundError):cached_loader()(manager)
            self.assertEqual(path.read_bytes(),original)
            self.assertFalse((shared/'y_4_raw.npy').exists())

if __name__=='__main__':unittest.main()
