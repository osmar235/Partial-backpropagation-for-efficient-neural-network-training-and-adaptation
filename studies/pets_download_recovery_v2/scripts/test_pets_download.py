import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
import download_pets_archives as d

class Tests(unittest.TestCase):
    def test_resume_retries(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'x';target.write_bytes(b'a');calls=[]
            def run(cmd,**kwargs):
                calls.append(cmd)
                with target.open('ab') as f:f.write(b'b' if len(calls)==1 else b'c')
                return subprocess.CompletedProcess(cmd,56 if len(calls)==1 else 0)
            d.download(target,hashlib.md5(b'abc').hexdigest(),['https://example.test/x'],'curl',run=run,sleep=lambda _:None)
            self.assertEqual(target.read_bytes(),b'abc');self.assertTrue(all('--continue-at' in c for c in calls))

    def test_resume_unsupported_preserves(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'x';target.write_bytes(b'partial');calls=[]
            def run(cmd,**kwargs):
                calls.append(cmd)
                if len(calls)==1:return subprocess.CompletedProcess(cmd,33)
                target.write_bytes(b'complete');return subprocess.CompletedProcess(cmd,0)
            d.download(target,hashlib.md5(b'complete').hexdigest(),['https://example.test/x'],'curl',run=run,sleep=lambda _:None)
            self.assertEqual(next(Path(td).glob('x.resume_unavailable.*')).read_bytes(),b'partial')
            self.assertNotIn('--continue-at',calls[1])

    def test_bad_checksum_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'x'
            def run(cmd,**kwargs):target.write_bytes(b'bad');return subprocess.CompletedProcess(cmd,0)
            with self.assertRaises(RuntimeError):d.download(target,hashlib.md5(b'good').hexdigest(),['https://example.test/x'],'curl',attempts=2,run=run,sleep=lambda _:None)
            self.assertEqual(len(list(Path(td).glob('x.checksum_failed.*'))),2)

    def test_safe_extraction_and_skip(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);archive=base/'a.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo('annotations/trainval.txt');info.size=3;tar.addfile(info,io.BytesIO(b'abc'))
            d.extract_verified(archive,d.md5(archive),'annotations',base)
            d.extract_verified(archive,d.md5(archive),'annotations',base)
            self.assertEqual((base/'annotations/trainval.txt').read_bytes(),b'abc')

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);archive=base/'a.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo('../escape');info.size=1;tar.addfile(info,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):d.extract_verified(archive,d.md5(archive),'images',base)
            self.assertFalse((base.parent/'escape').exists())

if __name__=='__main__':unittest.main()
