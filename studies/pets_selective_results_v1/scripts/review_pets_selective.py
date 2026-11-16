"""Read-only completed-development audit; writes a separate review folder."""
from pathlib import Path
import pets_selective as p
from analyze_pets_selective_results import analyze


def main():
    root=p.REAL;out=p.b.ROOT/'results/pets_selective_review_v1'
    assert not out.exists(),'Preserve existing reviews.'
    before={str(x):p.b.sha(x) for x in root.rglob('*') if x.is_file() and x.name!='RUN.lock'}
    device=p.torch.device('cuda');p.b.configure(device)
    cfg=p.configuration(False,False,device)
    assert cfg==p.b.read(root/'development_freeze.json')
    captured={};original_write=p.b.write
    def capture(path,value):
        assert Path(path)==root/'audit.json'
        captured.update(value)
    p.b.write=capture
    try:
        with p.b.lock(root):p.audit(root,device)
    finally:p.b.write=original_write
    assert captured==p.b.read(root/'audit.json')
    assert captured['integrity']=='PASS' and not captured['smoke'] and not captured['fragmented']
    assert len(captured['rows'])==18 and len(captured['rescores'])==36 and not captured['test_access']
    stats=analyze(root)
    for a,b in zip(stats['pairs'],captured['pairs']):
        assert a['seed']==b['seed'] and a['arm']==b['arm']
        assert abs(a['accuracy_difference_pp']-b['accuracy_difference_pp'])<1e-5
        assert abs(a['complete_saving_percent']-b['complete_saving_percent'])<1e-10
    after={str(x):p.b.sha(x) for x in root.rglob('*') if x.is_file() and x.name!='RUN.lock'}
    assert before==after
    out.mkdir()
    p.b.write(out/'analysis.json',stats)
    p.b.write(out/'development_inputs_sha256.json',before)
    p.b.write(out/'verification.json',dict(status='PASS',trajectories=18,seeds=6,unchanged_files=len(before),
        checkpoint_rescores=36,max_logit_error=max(x['max_error'] for x in captured['rescores']),
        fragmented=False,training_runs_this_review=0,test_access=False))
    print(p.b.json.dumps({k:stats[k] for k in ('means','comparisons','planning')},indent=2))


if __name__=='__main__':main()
