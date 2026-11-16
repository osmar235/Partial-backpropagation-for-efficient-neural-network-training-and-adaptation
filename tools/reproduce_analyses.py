"""Replay selected principal analyses without retraining or changing source evidence."""
from pathlib import Path
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from verify_package import verify

ROOT = Path(__file__).resolve().parents[1]

def main():
    integrity = verify()
    output_root = ROOT/'reproduced'
    output_root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='analysis_', dir=output_root))
    checks = []

    def run(name, folder, arguments):
        result = subprocess.run([sys.executable, '-B', *arguments], cwd=folder, text=True, encoding='utf8', capture_output=True)
        (out/(name+'.stdout.txt')).write_text(result.stdout, encoding='utf8')
        (out/(name+'.stderr.txt')).write_text(result.stderr, encoding='utf8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out}')
        checks.append(dict(name=name, status='PASS'))
        print(name+': PASS', flush=True)

    studies = ROOT/'studies'
    legacy = studies/'public_candidate_v1/legacy_stage6b'
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary)/'legacy'
        shutil.copytree(legacy, isolated)
        before = json.loads((legacy/'results/stage6b/stage6b_heldout_aggregate/summary.json').read_text(encoding='utf8'))
        run('cnn_confirmation', isolated, ['scripts/analyze_stage6b.py'])
        after = json.loads((isolated/'results/stage6b/stage6b_heldout_aggregate/summary.json').read_text(encoding='utf8'))
        if before != after:
            raise RuntimeError('CNN confirmation differs from archived summary')

    run('qkv_adaptation', studies/'public_candidate_v1', ['qkv/tools/analyze_training.py'])
    run('attention_reference', studies/'attention_reference_results_v1', ['recompute_timings.py'])
    for name in ['bf16_continuation', 'optimized_prefix_cost', 'prefix_adaptation']:
        capsule = studies/(name+'_results_v1')
        target = out/(name+'.json')
        run(name, capsule, ['analysis/summarize_results_v1.py', 'review_data.json', '--output', str(target)])
        expected = json.loads((capsule/'group_summary.json').read_text(encoding='utf8'))
        if json.loads(target.read_text(encoding='utf8')) != expected:
            raise RuntimeError(name+' differs from archived grouped statistics')
    for name, script in [
        ('pets_baseline','source/scripts/analyze_pets_baseline_results.py'),
        ('pets_selective','scripts/analyze_pets_selective_results.py'),
        ('pets_independent','scripts/reproduce_pets_evaluation.py'),
    ]:
        run(name, studies/(name+'_results_v1'), [script, '--input', 'evidence'])
    report = dict(status='PASS', integrity=integrity, checks=checks, python=sys.version,
                  scope='Analysis replay from included evidence; no training or checkpoint regeneration.')
    (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print('All selected analyses passed. Outputs: '+str(out))

if __name__ == '__main__':
    main()
