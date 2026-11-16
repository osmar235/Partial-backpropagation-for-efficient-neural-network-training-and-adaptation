"""Reconstruct diagnostic statistics from saved tensors without forward/backward."""
import argparse
from collections import Counter
from pathlib import Path
import torch
import optimizer_sensitivity as d

def audit(out,write=False):
    out=Path(out).resolve();cfg=d.read(out/'configuration.json');spec=cfg['spec'];reference=spec['reference']
    assert cfg['fingerprint']==d.e.q.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    for p,h in {**cfg['sources'],**cfg['evidence']}.items():assert d.e.q.d.sha(p)==h,p
    if not cfg['smoke']:assert cfg==d.read(d.FREEZE)
    if (out/'COMPLETE.json').exists():assert d.e.q.d.validate_complete(out,cfg['fingerprint'])
    rows=[];status=Counter();attempt_counts={};raw_bytes=0
    for task in d.trajectories(cfg):
        parent=out/task['key'];fp=d.e.q.v.digest(dict(root=cfg['fingerprint'],task=task))
        marker=d.e.q.d.validate_complete(parent,fp);assert marker
        attempt=Path(marker['attempt']).resolve();assert attempt.parent==parent
        assert set(marker['sha256'])=={str(p.resolve()) for p in attempt.rglob('*') if p.is_file()}
        process=d.read(attempt/'process.json');assert process['task']==task and process['root_fingerprint']==cfg['fingerprint']
        attempt_counts[task['key']]=len(list(parent.glob('attempt_*')))
        inputs=torch.load(attempt/'inputs.pt',map_location='cpu',weights_only=False)
        x,y,provenance=d.inputs(task,cfg,torch.device('cpu'))
        assert torch.equal(x,inputs['x']) and torch.equal(y,inputs['y']) and provenance==inputs['provenance']
        bridge=torch.load(attempt/'trajectory_bridge.pt',map_location='cpu',weights_only=False)
        source=d.read(attempt/'source.json')
        for anchor in spec['anchors']:
            rawpath=attempt/f'anchor_{anchor}.pt';raw_bytes+=rawpath.stat().st_size
            raw=torch.load(rawpath,map_location='cpu',weights_only=False);weights=raw['weights'];contexts=raw['contexts'];captures=raw['captures']
            assert raw['task']==task and raw['anchor']==anchor and raw['reference_repeat_exact']
            expected=['fresh']+(['archived'] if anchor==0 and task['kind']=='trained' else ['retained_two_updates'] if anchor==2 else [])
            assert list(contexts)==expected and list(captures)==spec['arms'] and list(raw['replays'])==expected
            if anchor==0 and task['kind']=='trained' and not cfg['smoke']:
                cp=torch.load(cfg['checkpoints'][str(task['seed'])]['path'],map_location='cpu',weights_only=False)
                assert d.e.q.old.state_hash(weights)==d.e.q.old.state_hash(cp['model_state'])
                assert d.e.q.v.digest(source['original_optimizer_groups'])==d.e.q.v.digest(cp['optimizer_state']['param_groups'])
                expectedstate=d.cpu(cp['optimizer_state']);expectedstate['param_groups']=d.cpu(contexts['fresh']['param_groups'])
                assert d.opt_hash(expectedstate)==d.opt_hash(contexts['archived'])
            if anchor==2:
                assert all(torch.equal(t,bridge['second_weights'][n]) for n,t in weights.items())
                assert d.opt_hash(contexts['retained_two_updates'])==bridge['second_step']['end_optimizer_sha256']
                assert bridge['second_step']['start_optimizer_sha256']==bridge['first_step']['end_optimizer_sha256']
                assert bridge['second_step']['start_model_sha256']==d.e.q.old.state_hash(bridge['first_weights'])
                assert all(torch.equal(t-bridge['first_weights'][n],bridge['second_step']['delta'][n]) for n,t in weights.items())
            for arm,cap in captures.items():
                status[cap['status']]+=1
                assert cap['status'] in ('FINITE','UNSUPPORTED','NONFINITE')
                if cap['status']!='FINITE':continue
                assert list(cap['gradient'])==list(weights)
                assert all(g.shape==weights[n].shape and g.dtype==torch.float32 and torch.isfinite(g).all() for n,g in cap['gradient'].items())
                assert sum(cap['dispatch'].values())==(2 if cfg['smoke'] else 6)
                flags=cap['policy'];assert flags['deterministic']==flags['cudnn_deterministic']==arm.endswith('_math')
                assert not any(flags[k] for k in ('tf32_matmul','tf32_cudnn','cudnn_benchmark','warn_only','fp16_reduced_reduction','bf16_reduced_reduction','math_reduced_reduction'))
                if arm.endswith('_math'):assert cap['dispatch']['math']==sum(cap['dispatch'].values())
            ref=captures[reference];assert ref['status']=='FINITE';summaries=[]
            for context,state in contexts.items():
                age=0 if context=='fresh' else 2 if context=='retained_two_updates' or cfg['smoke'] else 17600
                d.validate_state(state,weights,age)
                assert bool(state['state'])==(age!=0)
                group=state['param_groups'][0]
                assert group['lr']==spec['learning_rate'] and group['eps']==spec['epsilon'] and group['weight_decay']==spec['weight_decay'] and list(group['betas'])==spec['betas']
                records=raw['replays'][context];assert list(records)==[a for a in spec['arms'] if captures[a]['status']=='FINITE']
                for arm,record in records.items():
                    assert record['start_model_sha256']==d.e.q.old.state_hash(weights) and record['start_optimizer_sha256']==d.opt_hash(state)
                    assert record['optimizer_age']==age+1 and list(record['delta'])==list(weights)
                    assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in record['delta'].values())
                    summary=d.analyze(weights,state,ref['gradient'],captures[arm]['gradient'],records[reference]['delta'],record['delta'],spec)
                    if arm==reference:assert summary['update']['max_abs']==summary['gradient']['max_abs']==0
                    assert sum(z['coordinates'] for z in summary['bins'])==summary['coordinates']
                    assert sum(z['sign_flips'] for z in summary['bins'])==summary['strict_sign_flips']
                    summaries.append(dict(context=context,arm=arm,**summary))
                    rows.append(dict(task=task['key'],kind=task['kind'],seed=task['seed'],domain=task['domain'],anchor=anchor,context=context,arm=arm,
                        gradient_relative_l2=summary['gradient']['relative_l2'],gradient_cosine=summary['gradient']['cosine'],
                        update_relative_l2=summary['update']['relative_l2'],max_update_difference=summary['update']['max_abs'],
                        sign_flip_fraction=summary['sign_flip_fraction'],small_gradient_update_mass=sum(z['update_difference_mass_fraction'] for z in summary['bins'][:3])))
                if anchor==0 and context=='fresh':
                    assert records[reference]['end_optimizer_sha256']==bridge['first_step']['end_optimizer_sha256']
                    assert all(torch.equal(bridge['first_weights'][n]-t,records[reference]['delta'][n]) for n,t in weights.items())
            assert summaries==d.read(attempt/f'anchor_{anchor}_summary.json'),'Saved summary differs from raw-tensor reconstruction'
    result=dict(status='PASS',smoke=cfg['smoke'],trajectories=len(attempt_counts),anchors=2*len(attempt_counts),replay_comparisons=len(rows),
        capture_status_counts=dict(status),fragmented_trajectories=[k for k,v in attempt_counts.items() if v!=1],raw_anchor_bytes=raw_bytes,
        performance_measured=False,quality_established=False,historical_gate_retested=False)
    if write:
        d.save(out/'audit.json',result);d.save(out/'summary.json',rows)
        (out/'REPORT.md').write_text('# Optimizer sensitivity diagnostic\n\nAudit PASS: raw-tensor summaries reconstructed; no forward/backward rerun.\n'
            f'{len(rows)} replay comparisons across {result["anchors"]} anchors.\n'
            'Read summary.json and per-anchor summaries/raw evidence before interpretation.\n'
            'No timing or quality conclusion, and no historical gate relaxation.\n',encoding='utf8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();d.e.q.m.configure();print(audit(args.output))
