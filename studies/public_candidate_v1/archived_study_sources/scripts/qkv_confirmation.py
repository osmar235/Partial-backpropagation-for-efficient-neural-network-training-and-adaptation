"""Prospective session replication of unchanged QKV-unbind update timing."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import qkv_layout as e

ROOT=e.ROOT
HERE=ROOT/'protocol/qkv_confirmation'
SPEC=HERE/'confirmation_spec_v1.json'
FREEZE=HERE/'FROZEN_CONFIRMATION_v1.json'
OUT=ROOT/'results/qkv_confirmation_v1'

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def sources():
    names=['scripts/qkv_confirmation.py','scripts/review_qkv_confirmation.py',
           'scripts/test_qkv_confirmation.py','scripts/run_qkv_confirmation.cmd',
           'protocol/qkv_confirmation/confirmation_spec_v1.json',
           'protocol/qkv_confirmation/CONFIRMATION_PROTOCOL_v1.md']
    return {**e.sources(),**{str(ROOT/p):e.d.sha(ROOT/p) for p in names}}

def configuration(smoke,device):
    spec=read(SPEC)
    if smoke:spec.update(sessions=2,seeds=[99103],batch_sizes=[3,4],rounds=2,
                        discarded_warmup_updates=1,steady_updates=3)
    cfg=e.freeze(spec,smoke,device)
    cfg['sources']=sources();cfg.pop('fingerprint');cfg['fingerprint']=e.v.digest(cfg)
    return cfg

def schedule(spec,session):
    conditions=list(itertools.product(spec['seeds'],spec['batch_sizes']))
    if session>=spec['sessions']//2:conditions.reverse()
    shift=session%len(conditions)
    return conditions[shift:]+conditions[:shift]

def seal(folder,cfg,paths):
    e.v.atomic_json(folder/'COMPLETE.json',dict(fingerprint=cfg['fingerprint'],
        sha256={str(p):e.d.sha(p) for p in paths}))

def session_run(out,session,cfg,device):
    from review_qkv_confirmation import audit_session
    parent=out/f'session_{session:02d}';parent.mkdir(exist_ok=True)
    if e.d.validate_complete(parent,cfg['fingerprint']):
        audit_session(out,session,cfg);print('SKIP completed session',session,flush=True);return
    launch=e.d.next_attempt(parent/'launches')
    # Record process boundaries even when interrupted; never silently replace a session.
    e.v.atomic_json(launch/'process.json',dict(pid=os.getpid(),uuid=str(uuid.uuid4()),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    e.v.atomic_json(launch/'preflight.json',{'smoke':True} if cfg['smoke'] else e.v.idle_preflight())
    e.v.atomic_json(launch/'telemetry_before.json',{'smoke':True} if cfg['smoke'] else e.v.telemetry())
    for position,(seed,batch) in enumerate(schedule(cfg['spec'],session)):
        target=parent/f'seed_{seed}_batch_{batch}'
        # ci controls alternating arm order. Persist it in the inherited run config.
        ci=position+session
        e.condition(target,seed,batch,ci,cfg,device)
        e.v.atomic_json(launch/f'condition_{position:02d}.json',dict(seed=seed,batch=batch,
            telemetry={'smoke':True} if cfg['smoke'] else e.v.telemetry()))
    e.v.atomic_json(launch/'telemetry_after.json',{'smoke':True} if cfg['smoke'] else e.v.telemetry())
    audit_session(out,session,cfg,require_complete=False)
    paths=list(parent.glob('seed_*/COMPLETE.json'))+list(parent.glob('seed_*/run_config.json'))
    paths+=list((parent/'launches').rglob('*.json'))
    seal(parent,cfg,paths)
    print('SESSION COMPLETE',session,flush=True)

def main():
    ap=argparse.ArgumentParser()
    action=ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze-only',action='store_true');action.add_argument('--run-all',action='store_true')
    action.add_argument('--session',type=int)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--tag',default='v1');args=ap.parse_args()
    assert args.tag.replace('_','').isalnum()
    e.m.configure();device=e.torch.device(args.device)
    assert args.smoke or device.type=='cuda'
    assert device.type=='cpu' or e.torch.cuda.is_available()
    cfg=configuration(args.smoke,device)
    if args.freeze_only:
        assert not args.smoke and not OUT.exists()
        e.old.write_freeze(FREEZE,cfg);print('FROZEN',cfg['fingerprint']);return
    if not args.smoke:assert cfg==read(FREEZE),'Frozen sources/spec/environment changed'
    out=ROOT/f'results/qkv_confirmation_smoke_{args.tag}_{device.type}' if args.smoke else OUT
    out.mkdir(exist_ok=True);e.old.write_freeze(out/'configuration.json',cfg)
    from review_qkv_confirmation import audit
    if e.d.validate_complete(out,cfg['fingerprint']):
        print(json.dumps(audit(out),indent=2));return
    if args.session is not None:
        assert 0<=args.session<cfg['spec']['sessions']
        session_run(out,args.session,cfg,device);return
    for session in range(cfg['spec']['sessions']):
        print(f'START SESSION {session+1}/{cfg["spec"]["sessions"]}',flush=True)
        argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--session',str(session),'--device',args.device,'--tag',args.tag]
        if args.smoke:argv.append('--smoke')
        subprocess.run(argv,cwd=ROOT,check=True)
        # Let device telemetry settle between sessions; no speed-dependent stopping.
        if not args.smoke and session+1<cfg['spec']['sessions']:time.sleep(5)
    result=audit(out,write=True)
    paths=[out/p for p in ('configuration.json','audit.json','session_summary.json','REPORT.md')]
    paths+=list(out.glob('session_*/COMPLETE.json'))
    seal(out,cfg,paths)
    print(json.dumps(result,indent=2));print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
