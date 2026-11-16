"""Audit stored functional evidence; optimizer replays, no model forward/backward."""
import argparse
from collections import Counter
from pathlib import Path
import torch
import key_bias_function as f

def check_output(output,weights,policy,batch,cfg,status):
    status[output['status']]+=1
    assert output['status'] in ('FINITE','UNSUPPORTED','NONFINITE')
    if output['status']!='FINITE':
        if output['status']=='UNSUPPORTED':assert output['reason']
        return
    logits=output['logits'];assert logits.shape==(len(batch['y']),10) and logits.dtype==torch.float32
    assert torch.isfinite(logits).all() and output['model_sha256']==f.e.q.old.state_hash(weights)
    f.score(output,batch['y'],cfg['spec'])
    flags=output['policy'];assert flags['deterministic']==flags['cudnn_deterministic']==policy.endswith('_math')
    assert not any(flags[k] for k in ('tf32_matmul','tf32_cudnn','cudnn_benchmark','warn_only','fp16_reduced_reduction','bf16_reduced_reduction','math_reduced_reduction'))
    assert sum(output['dispatch'].values())==(2 if cfg['smoke'] else 6)
    if policy.endswith('_math'):assert output['dispatch']['math']==sum(output['dispatch'].values())

def verify_swaps(reference,candidate,components):
    # Independent per-coordinate donor masks, including Q/V of packed biases.
    count=0
    for name,ref in reference.items():
        mask=torch.zeros_like(ref,dtype=torch.bool)
        if name.endswith('self_attention.in_proj_bias'):
            width=ref.numel()//3;mask[width:2*width]=True;count+=width
        assert torch.equal(components['key_only'][name],torch.where(mask,candidate[name],ref))
        assert torch.equal(components['key_restored'][name],torch.where(mask,ref,candidate[name]))
    return count

def audit(out,write=False):
    out=Path(out).resolve();cfg=f.read(out/'configuration.json');device=torch.device(cfg['device'])
    assert cfg['fingerprint']==f.e.q.v.digest({k:v for k,v in cfg.items() if k!='fingerprint'})
    for p,h in {**cfg['sources'],**cfg['evidence']}.items():assert f.e.q.d.sha(p)==h,p
    if not cfg['smoke']:assert cfg==f.read(f.FREEZE)
    if (out/'COMPLETE.json').exists():assert f.e.q.d.validate_complete(out,cfg['fingerprint'])
    rows=[];runtime=[];repeats=[];status=Counter();attempts={};replays=cases=0
    for record in cfg['records']:
        parent=out/record['key'];fp=f.e.q.v.digest(dict(root=cfg['fingerprint'],record=record))
        marker=f.e.q.d.validate_complete(parent,fp);assert marker
        attempt=Path(marker['attempt']).resolve();assert attempt.parent==parent
        assert set(marker['sha256'])=={str(p.resolve()) for p in attempt.rglob('*') if p.is_file()}
        process=f.read(attempt/'process.json');assert process['root_fingerprint']==cfg['fingerprint'] and process['record']==record
        attempts[record['key']]=len(list(parent.glob('attempt_*')))
        raw=torch.load(record['raw_path'],map_location='cpu',weights_only=False)
        assert raw['anchor']==record['anchor'] and raw['task']['seed']==record['seed']
        packet=torch.load(attempt/'outputs.pt',map_location='cpu',weights_only=False);assert packet['record']==record
        data=f.batches(record,cfg);assert list(packet['batches'])==list(data)==list(packet['unchanged'])
        for name,batch in data.items():
            saved=packet['batches'][name]
            assert torch.equal(saved['x'],batch['x']) and torch.equal(saved['y'],batch['y']) and saved['provenance']==batch['provenance']
            assert list(packet['unchanged'][name])==cfg['spec']['evaluation_policies']
            for policy,output in packet['unchanged'][name].items():check_output(output,raw['weights'],policy,batch,cfg,status)
        assert list(packet['contexts'])==list(raw['contexts'])
        for context,case in packet['contexts'].items():
            states,proof=f.recovered(raw,context,cfg,device);assert proof==case['proof'];replays+=len(states)
            reference=states['reference_fp32_math'];variants={}
            eligible=[a for a in cfg['spec']['candidate_arms'] if a in states]
            assert list(case['swaps'])==eligible and list(case['outputs'])==list(data)
            for arm in eligible:
                before=[f.e.q.old.state_hash(s) for s in (reference,states[arm])]
                parts,count=f.swapped(reference,states[arm]);assert verify_swaps(reference,states[arm],parts)==count==(48 if cfg['smoke'] else 1152)
                assert before==[f.e.q.old.state_hash(s) for s in (reference,states[arm])]
                variants[arm]=dict(candidate=states[arm],**parts)
                assert case['swaps'][arm]==dict(key_coordinates=count,hashes={k:f.e.q.old.state_hash(v) for k,v in variants[arm].items()});cases+=1
            for name,policies in case['outputs'].items():
                assert list(policies)==cfg['spec']['evaluation_policies']
                for policy,block in policies.items():
                    assert set(block)=={'reference','repeat','candidates'} and list(block['candidates'])==eligible
                    for key in ('reference','repeat'):check_output(block[key],reference,policy,data[name],cfg,status)
                    for arm,group in block['candidates'].items():
                        assert list(group)==list(variants[arm])
                        for variant,output in group.items():check_output(output,variants[arm][variant],policy,data[name],cfg,status)
        summary=f.summarize(packet,cfg);assert summary==f.read(attempt/'summary.json')
        for key,dest in (('comparisons',rows),('runtime_reference_differences',runtime),('repeat_differences',repeats)):
            dest.extend(dict(record=record['key'],seed=record['seed'],anchor=record['anchor'],**r) for r in summary[key])
    expected_records=len(cfg['spec']['trained_seeds'])*len(cfg['spec']['anchors']);assert len(attempts)==expected_records
    result=dict(status='PASS',smoke=cfg['smoke'],anchors=len(attempts),component_cases=cases,output_status_counts=dict(status),
        optimizer_replays_during_audit=replays,within_policy_comparisons=len(rows),reference_precision_comparisons=len(runtime),repeat_checks=len(repeats),
        fragmented_conditions=[k for k,v in attempts.items() if v!=1],model_forwards_during_audit=0,backwards_during_audit=0,
        quality_established=False,performance_measured=False,historical_gates_changed=False)
    if not cfg['smoke']:
        assert cases==40 and replays==48 and sum(status.values())==840
        if status['FINITE']==840:assert len(rows)==240 and len(runtime)==len(repeats)==48
    if write:
        f.save(out/'audit.json',result);f.save(out/'summary.json',dict(comparisons=rows,runtime_reference_differences=runtime,repeat_differences=repeats))
        (out/'REPORT.md').write_text('# Key-bias functional diagnostic\n\nIntegrity audit PASS. Saved-gradient optimizer steps replayed on cloned parameters; no model forward/backward during audit.\n'
            f'{len(rows)} eligible within-policy comparisons; output statuses: {dict(status)}.\n'
            'Interpret saved effects with precision and repeat controls. Unsupported/nonfinite outputs cannot support conclusions.\n'
            'No long-run quality, performance, or historical gate verdict follows from this diagnostic.\n',encoding='utf8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();f.e.q.m.configure();print(audit(args.output))
