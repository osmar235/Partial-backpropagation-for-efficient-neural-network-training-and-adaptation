"""Read-only validation of local Stage 7 integration and archival integrity."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    with (ROOT/'protocol/stage7/frozen_files_before.csv').open(encoding='utf-8-sig',newline='') as f:
        original=list(csv.DictReader(f))
    for row in original:
        assert hashlib.sha256(Path(row['Path']).read_bytes()).hexdigest().upper()==row['Hash'], row['Path']
    print(f'OK: {len(original)} frozen Stage 6/release files unchanged')
    name='archived_epoch_q_summary.csv'
    def records(path):
        with path.open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    actual=records(ROOT/'results/stage7A_original_audit'/name)
    supplied=records(ROOT/'results/stage7A_supplied_reference'/name)
    assert len(actual)==len(supplied)==30
    for a,b in zip(actual,supplied):
        for k in a:
            assert math.isclose(float(a[k]),float(b[k]),rel_tol=1e-10,abs_tol=1e-10), (k,a,b)
    print('OK: all 30 archived audit rows agree with supplied reference')
    for folder in ('stage7_smoke_v1_1','stage7_cuda_integration_v1_1'):
        summaries=list((ROOT/'results'/folder).glob('seed_*/*/summary.json'))
        assert len(summaries)==11
        for path in summaries:
            r=json.loads(path.read_text())
            assert r['full_updates']+r['partial_updates']==r['updates']
            assert math.isclose(r['backward_full_seconds']+r['backward_partial_seconds'],r['backward_seconds'])
            assert r['status']=='complete' and len(r['config_fingerprint'])==64
            if r['variant']!='no_forced_full': assert r['max_consecutive_partials']<=1
    print('OK: CPU/CUDA completion records, mode counts and timing totals')
    config_path=ROOT/'results/stage7_smoke_v1_1/seed_050/run_config.json'
    config=json.loads(config_path.read_text())
    command=[sys.executable,'-B',str(ROOT/'scripts/stage7_review_experiments.py')]
    for key in ('suite','dataset','train_subset','eval_subset','batch_size','epochs',
                'schedule_epochs','warmup_steps','seed','device','out'):
        command += ['--'+key.replace('_','-'),str(config[key])]
    command += ['--tiny-model','--log-steps','--variants',*config['variants']]
    before=config_path.read_bytes()
    resumed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    assert resumed.returncode==0, resumed.stderr
    assert resumed.stdout.count('[SKIP]')==11, resumed.stdout
    mismatch=subprocess.run(command+['--lr','0.0004'],cwd=ROOT,capture_output=True,text=True)
    assert mismatch.returncode!=0 and 'different settings/code/environment' in mismatch.stderr, mismatch.stderr
    assert config_path.read_bytes()==before
    incomplete=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/stage7_summarize.py'),
        '--root',config['out'],'--expected-seeds','50','51','--expected-variants',*config['variants']],
        cwd=ROOT,capture_output=True,text=True)
    assert incomplete.returncode!=0 and 'Incomplete or unexpected study' in incomplete.stderr
    print('OK: identical resume skips all variants; changed settings and incomplete studies rejected')

if __name__=='__main__':
    main()
