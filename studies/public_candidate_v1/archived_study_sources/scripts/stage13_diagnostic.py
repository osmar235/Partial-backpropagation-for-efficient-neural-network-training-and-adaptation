"""Bounded local-objective diagnostic at preserved Stage12 head-warmup states."""
import argparse,copy,json,math,platform,re,time
from pathlib import Path
import torch
import torch.nn.functional as F
import stage13_local as l
import stage12_adaptation as previous
p=l.p;v=p.v;d=previous.d;old=previous.old
ROOT=v.ROOT;SPEC=ROOT/'protocol/stage13/local_diagnostic_spec_v1.json'
FREEZE=ROOT/'protocol/stage13/FROZEN_LOCAL_DIAGNOSTIC_v1.json';REAL=ROOT/'results/stage13A_local_diagnostic_v1'
RAW=ROOT/'results/stage12A_adaptation_v1'

def sources():
    names=['scripts/stage13_local.py','scripts/stage13_diagnostic.py','scripts/review_stage13A.py','scripts/test_stage13_local.py',
           'scripts/run_stage13A_local_diagnostic.cmd','protocol/stage13/local_diagnostic_spec_v1.json','protocol/stage13/LOCAL_DIAGNOSTIC_v1.md']
    return {**previous.sources(),**{str(ROOT/n):p.sha(ROOT/n) for n in names}}

def freeze_record(spec,fake):
    evidence={};checkpoints={};split={'kind':'artificial'}
    if not fake:
        cfg=json.loads((RAW/'configuration.json').read_text());assert not cfg['synthetic'];d.validate_complete(RAW,cfg['fingerprint'])
        split=cfg['split']
        for seed in spec['source_seeds']:
            parent=RAW/f'seed_{seed}'/spec['source_arm'];rc=json.loads((parent/'run_config.json').read_text())
            marker=d.validate_complete(parent,rc['fingerprint']);path=Path(marker['attempt'])/spec['source_checkpoint']
            checkpoints[str(seed)]=dict(path=str(path),sha256=p.sha(path),root_fingerprint=cfg['fingerprint'],fingerprint=rc['fingerprint'])
        for rel in ['results/stage12A_adaptation_v1/configuration.json','results/stage12A_adaptation_v1/COMPLETE.json',
                    'results/stage12A_review_v1/REPORT.md','results/stage12A_review_v1/independent_audit.json']:
            evidence[str(ROOT/rel)]=p.sha(ROOT/rel)
    record=dict(date='2026-10-22',spec=spec,synthetic=fake,split=split,checkpoints=checkpoints,evidence_sha256=evidence,
        source_sha256=sources(),required_execution=p.execution.execution_settings(),
        pretrained=None if fake else dict(path=str(p.WEIGHTS),sha256=p.sha(p.WEIGHTS)))
    record['freeze_hash']=v.digest(record);return record

def data(device,fake):
    if fake:
        gen=torch.Generator().manual_seed(313001)
        return torch.rand(60,3,32,32,generator=gen).to(device),torch.arange(10,device=device).repeat_interleave(6),{'kind':'artificial'}
    x,y,ev,ey,split=p.load_data(device,False);del ev,ey;return x,y,split

def streams(y,seed,spec):
    gen=torch.Generator().manual_seed(seed+314001);cpu=y.cpu();pools={key:[] for key in ['calibration','probe','score']}
    for cls in range(10):
        ix=torch.where(cpu==cls)[0];ix=ix[torch.randperm(len(ix),generator=gen)];offset=0
        for role in pools:
            count=spec[role+'_per_class'];assert len(ix)>=offset+count
            pools[role]+=ix[offset:offset+count].tolist();offset+=count
    result=dict(pools=pools,calibration_batches=[],probe_batches=[],score_batches=[])
    for i in range(spec['calibration_steps']):
        result['calibration_batches'].append([pools['calibration'][j] for j in torch.randint(len(pools['calibration']),(spec['batch_size'],),generator=gen).tolist()])
    for i in range(spec['probe_trials']):
        for role in ['probe','score']:
            assert len(pools[role])>=spec['batch_size']
            result[role+'_batches'].append([pools[role][j] for j in torch.randperm(len(pools[role]),generator=gen)[:spec['batch_size']].tolist()])
    return result

def batch(x,y,ix,size,augment_seed=None):
    index=torch.tensor(ix,device=x.device)
    return p.transform(x[index],size,None if augment_seed is None else torch.Generator().manual_seed(augment_seed)),y[index]

def source(seed,fake,record,x,y,spec,device):
    net=p.make_model(seed,fake).to(device)
    if fake:
        opt=p.optimizer(net,dict(backbone_lr=1e-4,head_lr=.001,weight_decay=.01))
        xx=p.transform(x[:3],spec['image_size'])
        for _ in range(2):p.update(net,opt,xx,y[:3],'head',0.)
    else:
        entry=record['checkpoints'][str(seed)];cp=torch.load(entry['path'],map_location='cpu',weights_only=True)
        assert cp['seed']==seed and cp['arm']==spec['source_arm'] and cp['updates']==704
        assert cp['root_fingerprint']==entry['root_fingerprint'] and cp['fingerprint']==entry['fingerprint']
        net.load_state_dict(cp['model_state'],strict=True)
    net.eval();return net

def run_seed(attempt,seed,data_values,cfg,record,spec,device,fake):
    x,y,_=data_values;net=source(seed,fake,record,x,y,spec,device);source_hash=old.state_hash(net.state_dict())
    stream=streams(y,seed,spec);v.atomic_json(attempt/f'streams_{seed}.json',stream)
    v.sync(device);overhead_start=time.perf_counter();model=l.LocalModel(net,seed,fake).to(device)
    auxopt=torch.optim.AdamW(list(model.aux0.parameters())+list(model.aux1.parameters()),lr=spec['calibration_lr'],weight_decay=spec['weight_decay'])
    calibration=[]
    for i,ix in enumerate(stream['calibration_batches']):
        xx,yy=batch(x,y,ix,spec['image_size'],seed+315001+i);auxopt.zero_grad(set_to_none=True)
        with torch.no_grad():z0=model.block0(xx);z1=model.block1(z0)
        a=F.cross_entropy(model.auxiliary(z0,0),yy);b=F.cross_entropy(model.auxiliary(z1,1),yy)
        (a+b).backward();auxopt.step()
        calibration.append(dict(seed=seed,step=i+1,loss0=float(a.detach()),loss1=float(b.detach())))
    assert old.state_hash(net.state_dict())==source_hash
    model.zero_grad(set_to_none=True)
    initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
    old.checkpoint(attempt/f'calibrated_{seed}.pt',dict(seed=seed,fingerprint=cfg['fingerprint'],source_hash=source_hash,
        model_state=initial,aux_optimizer_state=auxopt.state_dict(),calibration_steps=spec['calibration_steps']))
    v.write_csv(attempt/f'calibration_{seed}.csv',calibration);v.sync(device);calibration_seconds=time.perf_counter()-overhead_start
    del auxopt
    zero_states={};cold=[]
    xx,yy=batch(x,y,stream['probe_batches'][0],spec['image_size'],seed+316001)
    for mode in spec['modes']:
        opt=l.optimizer(model,spec)
        for i in range(spec['warmup_steps_per_mode']):
            model.load_state_dict(initial);model.zero_grad(set_to_none=True);l.zero_optimizer(opt)
            v.sync(device);t=time.perf_counter();losses=l.step(model,opt,xx,yy,mode);v.sync(device);seconds=time.perf_counter()-t
            l.verify_step(model,opt,initial,mode)
            if i==0:cold.append(dict(seed=seed,mode=mode,seconds=seconds))
        l.zero_optimizer(opt)
        zero_states[mode]=copy.deepcopy(opt.state_dict())
        for state in zero_states[mode]['state'].values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.cpu()
        del opt
    rows=[];align=[];telemetry=[];first_states={}
    for trial in range(spec['probe_trials']):
        xx,yy=batch(x,y,stream['probe_batches'][trial],spec['image_size'],seed+316001+trial)
        score_x,score_y=batch(x,y,stream['score_batches'][trial],spec['image_size'])
        model.load_state_dict(initial);model.zero_grad(set_to_none=True)
        with torch.no_grad():before=F.cross_entropy(model(score_x),score_y).item()
        order=spec['modes'][trial%4:]+spec['modes'][:trial%4]
        if (trial//4)%2:order=list(reversed(order))
        gradients={}
        if device.type=='cuda' and trial%4==0:telemetry.append(dict(seed=seed,trial=trial,sample=json.dumps(v.telemetry())))
        for position,mode in enumerate(order):
            model.load_state_dict(initial);model.zero_grad(set_to_none=True);opt=l.optimizer(model,spec);opt.load_state_dict(copy.deepcopy(zero_states[mode]))
            v.sync(device)
            baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else 0
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            t=time.perf_counter();losses=l.step(model,opt,xx,yy,mode);v.sync(device);seconds=time.perf_counter()-t
            peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
            gradients[mode]=l.gradient_vectors(model);l.verify_step(model,opt,initial,mode)
            with torch.no_grad():after=F.cross_entropy(model(score_x),score_y).item()
            assert math.isfinite(before) and math.isfinite(after) and seconds>0
            row=dict(seed=seed,trial=trial,mode=mode,order_position=position,seconds=seconds,
                baseline_allocated_bytes=baseline,peak_allocated_bytes=peak,incremental_peak_bytes=peak-baseline,
                update_loss_sum=sum(losses),final_head_training_loss=losses[-1],score_loss_before=before,score_loss_after=after,
                score_improvement=before-after,optimizer_age=1)
            rows.append(row)
            if trial==0:
                old.checkpoint(attempt/f'one_step_{seed}_{mode}.pt',dict(seed=seed,mode=mode,fingerprint=cfg['fingerprint'],
                    model_state=model.state_dict(),optimizer_state=opt.state_dict()))
            del opt
        for mode in spec['modes']:
            for block in ['block0','block1','block2']:
                a=gradients[mode][block].double();b=gradients['full'][block].double()
                aa=float(torch.dot(a,a));bb=float(torch.dot(b,b));ab=float(torch.dot(a,b));den=math.sqrt(aa*bb)
                align.append(dict(seed=seed,trial=trial,mode=mode,block=block,dot=ab,norm_squared=aa,
                    full_norm_squared=bb,cosine=ab/den if den>0 else '',defined=den>0))
        del gradients
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);assert old.state_hash(model.net.state_dict())==source_hash
    v.write_csv(attempt/f'measurements_{seed}.csv',rows);v.write_csv(attempt/f'alignment_{seed}.csv',align)
    v.write_csv(attempt/f'cold_{seed}.csv',cold)
    if telemetry:v.write_csv(attempt/f'telemetry_{seed}.csv',telemetry)
    v.atomic_json(attempt/f'seed_{seed}.json',dict(seed=seed,fingerprint=cfg['fingerprint'],source_hash=source_hash,
        calibrated_hash=old.state_hash(initial),calibration_overhead_seconds=calibration_seconds,
        measured_operations=len(rows),probe_trials=spec['probe_trials'],source_backbone_unchanged=True))
    del model,net

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--smoke-tag',default='final_cpu');ap.add_argument('--smoke-device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,32}',args.smoke_tag);p.execution.configure();spec=json.loads(SPEC.read_text());fake=args.smoke
    if fake:spec.update(source_seeds=[9951,9952],image_size=32,batch_size=3,calibration_per_class=2,probe_per_class=2,score_per_class=2,
        calibration_steps=4,warmup_steps_per_mode=1,probe_trials=4)
    record=freeze_record(spec,fake)
    if args.freeze_only:
        assert not fake and not REAL.exists();old.write_freeze(FREEZE,record);print('FROZEN',record['freeze_hash']);return
    out=ROOT/'results'/('stage13_sanity_local_v1_'+args.smoke_tag) if fake else REAL
    if not fake:assert FREEZE.exists() and json.loads(FREEZE.read_text())==record,'Source/evidence freeze changed'
    out.mkdir(exist_ok=True)
    if fake:old.write_freeze(out/'SYNTHETIC_FREEZE.json',record)
    device=torch.device(args.smoke_device if fake else 'cuda')
    if device.type=='cuda':assert torch.cuda.is_available()
    cfg=dict(spec=spec,synthetic=fake,freeze_hash=record['freeze_hash'],source_sha256=record['source_sha256'],split=record['split'],
        checkpoints=record['checkpoints'],execution=p.execution.execution_settings(),device=str(device),
        python=platform.python_version(),torch=torch.__version__,torchvision=__import__('torchvision').__version__,
        numpy=__import__('numpy').__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu')
    cfg['fingerprint']=v.digest(cfg);cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:v.atomic_json(cp,cfg)
    from review_stage13A import audit
    if (out/'COMPLETE.json').exists():audit(out,False);print('Completed diagnostic verified read-only');return
    attempt=d.next_attempt(out);v.atomic_json(attempt/'preflight.json',v.idle_preflight() if device.type=='cuda' else [])
    start=time.perf_counter();values=data(device,fake);assert values[-1]==record['split']
    v.atomic_json(attempt/'data_loading.json',dict(seconds=time.perf_counter()-start))
    for seed in spec['source_seeds']:
        print('LOCAL DIAGNOSTIC source seed',seed,flush=True);run_seed(attempt,seed,values,cfg,record,spec,device,fake)
    files={str(f):p.sha(f) for f in attempt.iterdir() if f.is_file()}
    v.atomic_json(out/'RAW_COMPLETE.json',dict(fingerprint=cfg['fingerprint'],attempt=str(attempt),sha256=files))
    audit(out,True)
    files=[out/n for n in ['configuration.json','RAW_COMPLETE.json','summary.csv','audit.json','REPORT.md']]
    v.atomic_json(out/'COMPLETE.json',dict(status='complete',fingerprint=cfg['fingerprint'],sha256={str(f):p.sha(f) for f in files}))
    print('COMPLETE. Send',out,'for review.')

if __name__=='__main__':main()
