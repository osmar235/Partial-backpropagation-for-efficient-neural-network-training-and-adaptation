import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
import download_pets_archives_v2 as v2
from test_pets_download import Tests as RecoveryTests

class DeadlineTests(unittest.TestCase):
    def test_command_has_no_overall_deadline_but_retains_stall_protection(self):
        args=v2.command('curl','https://example.test/x',Path('x'),True)
        self.assertEqual(args[1],'--disable')
        self.assertEqual(args[args.index('--max-time')+1],'0')
        self.assertEqual(args[args.index('--connect-timeout')+1],'20')
        self.assertEqual(args[args.index('--speed-time')+1],'90')
        self.assertEqual(args[args.index('--speed-limit')+1],'1024')
        self.assertNotIn('--insecure',args)
        self.assertIn('--continue-at',args)

    def test_simulated_transfer_longer_than_an_hour_completes(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'archive';calls=[]
            def transport(args,**kwargs):
                calls.append(args);limit=float(args[args.index('--max-time')+1])
                # Synthetic transport models a successful 4,800-second transfer.
                if limit and limit<4800:
                    target.write_bytes(b'partial');return subprocess.CompletedProcess(args,28)
                target.write_bytes(b'verified');return subprocess.CompletedProcess(args,0)
            v2.engine.download(target,hashlib.md5(b'verified').hexdigest(),['https://example.test/x'],'curl',
                               attempts=2,run=transport,sleep=lambda _:None)
            self.assertEqual(len(calls),1)
            self.assertEqual(target.read_bytes(),b'verified')

if __name__=='__main__':unittest.main()
