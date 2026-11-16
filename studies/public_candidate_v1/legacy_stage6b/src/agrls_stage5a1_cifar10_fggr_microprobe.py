from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets
from torchvision.models import resnet18

import agrls_stage1_mnist_fullbatch_v081 as v081
import agrls_stage4a_minibatch_locked_v012 as v012
import agrls_stage4g_forward_gated_refresh as v4g


# Stage 5A1: CIFAR-10 / ResNet-18-GN transfer of Forward-Gated Gradient Refresh.
# Probe size is NOT tuned on CIFAR outcomes. Stage 5A0 used synthetic inputs only
# and a frozen timing-only rule: choose the largest tested probe whose calibrated
# break-even acceptance rate <= 0.30. That rule selected probe_size=128.
#
# Algorithms:
#   Adam tensor-wise:       every update uses full-batch F+B.
#   Periodic K=2 no-query:  alternate fresh F+B and blind reuse of last Adam
#                           displacement, with zero NN query on the reuse update.
#   FGGR microprobe:        when reuse-eligible, compare current vs stale-trial
#                           loss on the SAME 128-example subset of the new batch.
#                           Accept -> apply stale displacement, skip full F+B.
#                           Reject -> restore live parameters, pay full F+B and
#                           refresh Adam. Max one consecutive skipped backward.
#
# No loss comparison crosses mini-batch boundaries. GroupNorm is used so probe
# forwards do not change running normalization state.

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def replace_batchnorm_with_groupnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            groups = min(32, c)
            while c % groups != 0 and groups > 1:
                groups //= 2
            setattr(module, name, nn.GroupNorm(groups, c))
        else:
            replace_batchnorm_with_groupnorm(child)


def make_cifar_resnet18_gn() -> nn.Module:
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    replace_batchnorm_with_groupnorm(model)
    return model


def _subset_indices(n: int, k: int, seed: int) -> torch.Tensor:
    if k <= 0 or k > n:
        raise ValueError(f'subset size {k} must be in [1, {n}]')
    if k == n:
        return torch.arange(n)
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randperm(n, generator=g)[:k]


def _normalize_cifar(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def load_cifar10_tensors(root: Path, train_subset: int, eval_subset: int, seed: int, device: torch.device):
    root.mkdir(parents=True, exist_ok=True)
    tr = datasets.CIFAR10(root=str(root), train=True, download=True)
    te = datasets.CIFAR10(root=str(root), train=False, download=True)

    xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    ytr = torch.tensor(tr.targets, dtype=torch.long)
    xev = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    yev = torch.tensor(te.targets, dtype=torch.long)

    it = _subset_indices(len(xtr), train_subset, seed + 51001)
    ie = _subset_indices(len(xev), eval_subset, seed + 52001)
    xtr = _normalize_cifar(xtr.index_select(0, it)); ytr = ytr.index_select(0, it)
    xev = _normalize_cifar(xev.index_select(0, ie)); yev = yev.index_select(0, ie)
    return xtr.to(device), ytr.to(device), xev.to(device), yev.to(device)


def load_fake_cifar(train_subset: int, eval_subset: int, seed: int, device: torch.device):
    g = torch.Generator(device='cpu').manual_seed(seed + 53001)
    xtr = torch.randn(train_subset, 3, 32, 32, generator=g)
    ytr = torch.randint(0, 10, (train_subset,), generator=g)
    xev = torch.randn(eval_subset, 3, 32, 32, generator=g)
    yev = torch.randint(0, 10, (eval_subset,), generator=g)
    return xtr.to(device), ytr.to(device), xev.to(device), yev.to(device)


@torch.no_grad()
def evaluate_batched(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module, batch_size: int = 512):
    model.eval()
    n = int(x.shape[0]); loss_sum = 0.0; correct = 0
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]; yb = y[start:start + batch_size]
        logits = model(xb)
        loss_sum += float(loss_fn(logits, yb).item()) * int(xb.shape[0])
        correct += int((logits.argmax(dim=1) == yb).sum().item())
    model.train()
    return loss_sum / max(n, 1), correct / max(n, 1)


def timed_probe_forward(model, xb, yb, loss_fn, device):
    v081.sync(device); t0 = time.perf_counter()
    with torch.no_grad():
        val = float(loss_fn(model(xb), yb).item())
    v081.sync(device)
    return val, time.perf_counter() - t0


@torch.no_grad()
def add_displacement_(model: nn.Module, displacement: Dict[str, torch.Tensor], alpha: float = 1.0) -> None:
    for n, p in v4g.named_trainable(model):
        p.add_(displacement[n], alpha=alpha)


def evaluate_summary(
    model, xtr, ytr, xev, yev, loss_fn, eval_batch_size,
    label: str, updates: int,
    full_fpasses: int, probe_fpasses: int, bpasses: int,
    full_fsec: float, probe_fsec: float, bsec: float, wall: float,
    pure_full_f: float, pure_probe_f: float, pure_b: float,
    examples_forwarded: int, examples_backwarded: int,
    reuse_attempts: int, reuse_accepts: int, reuse_rejects: int,
    forced_refreshes: int,
):
    tr_loss, tr_acc = evaluate_batched(model, xtr, ytr, loss_fn, eval_batch_size)
    ev_loss, ev_acc = evaluate_batched(model, xev, yev, loss_fn, eval_batch_size)
    calibrated = full_fpasses * pure_full_f + probe_fpasses * pure_probe_f + bpasses * pure_b
    return {
        'algorithm': label,
        'final_train_loss': tr_loss,
        'final_train_accuracy': tr_acc,
        'final_eval_loss': ev_loss,
        'final_eval_accuracy': ev_acc,
        'accepted_updates': updates,
        'full_forward_passes': full_fpasses,
        'probe_forward_passes': probe_fpasses,
        'backward_passes': bpasses,
        'forward_seconds': full_fsec + probe_fsec,
        'full_forward_seconds': full_fsec,
        'probe_forward_seconds': probe_fsec,
        'backward_seconds': bsec,
        'nn_compute_seconds': full_fsec + probe_fsec + bsec,
        'wall_seconds': wall,
        'calibrated_nn_compute_seconds': calibrated,
        'examples_forwarded': examples_forwarded,
        'examples_backwarded': examples_backwarded,
        'reuse_attempts': reuse_attempts,
        'reuse_accepts': reuse_accepts,
        'reuse_rejects': reuse_rejects,
        'reuse_accept_rate': reuse_accepts / max(reuse_attempts, 1),
        'forced_refreshes': forced_refreshes,
        'updates_per_backward': updates / max(bpasses, 1),
        'implementation_revision': 'stage5a1-tensorwise-inplace-probe-trial',
    }


def run_adam(model, init_state, xtr, ytr, xev, yev, loss_fn, device, updates, batch_size, seed, lr,
             F, Fp, B, eval_batch_size):
    model.load_state_dict(init_state); model.to(device); model.train()
    state = v4g.init_adam_state(model)
    stream = v012.make_batch_stream(int(xtr.shape[0]), batch_size, seed)
    rows: List[dict] = []; ff = pf = bp = 0; ffs = pfs = bs = 0.0; exf = exb = 0; adam_t = 0
    v081.sync(device); w0 = time.perf_counter()
    for u in range(1, updates + 1):
        idx = next(stream); xb, yb = v012.batch_from_indices(xtr, ytr, idx, device)
        loss, f0, dtf = v4g.timed_forward_graph(model, xb, yb, loss_fn, device)
        dtb = v4g.timed_backward(loss, device)
        ff += 1; bp += 1; ffs += dtf; bs += dtb; exf += int(xb.shape[0]); exb += int(xb.shape[0]); adam_t += 1
        disp = v4g.adam_displacement_from_current_grads(model, state, adam_t, lr)
        v4g.apply_displacement(model, disp)
        rows.append({'algorithm':'adam_tensorwise','update':u,'event':'fresh_adam','batch_loss':f0,
                     'full_forward_passes':ff,'probe_forward_passes':pf,'backward_passes':bp,
                     'calibrated_nn_compute_seconds':ff*F+pf*Fp+bp*B})
    v081.sync(device); wall = time.perf_counter() - w0
    return rows, evaluate_summary(model,xtr,ytr,xev,yev,loss_fn,eval_batch_size,'adam_tensorwise',updates,
                                  ff,pf,bp,ffs,pfs,bs,wall,F,Fp,B,exf,exb,0,0,0,updates)


def run_periodic_noquery(model, init_state, xtr, ytr, xev, yev, loss_fn, device, updates, batch_size, seed, lr,
                         F, Fp, B, eval_batch_size):
    model.load_state_dict(init_state); model.to(device); model.train()
    state = v4g.init_adam_state(model)
    stream = v012.make_batch_stream(int(xtr.shape[0]), batch_size, seed)
    rows=[]; ff=pf=bp=0; ffs=pfs=bs=0.0; exf=exb=0; adam_t=0
    last_disp: Optional[Dict[str,torch.Tensor]] = None; reuse_next=False; accepts=0; fresh=0
    v081.sync(device); w0=time.perf_counter()
    for u in range(1,updates+1):
        idx=next(stream)
        if reuse_next and last_disp is not None:
            v4g.apply_displacement(model,last_disp); accepts+=1; reuse_next=False
            event='blind_periodic_reuse_no_query'; batch_loss=float('nan')
        else:
            xb,yb=v012.batch_from_indices(xtr,ytr,idx,device)
            loss,f0,dtf=v4g.timed_forward_graph(model,xb,yb,loss_fn,device); dtb=v4g.timed_backward(loss,device)
            ff+=1; bp+=1; ffs+=dtf; bs+=dtb; exf+=int(xb.shape[0]); exb+=int(xb.shape[0]); adam_t+=1
            last_disp=v4g.adam_displacement_from_current_grads(model,state,adam_t,lr); v4g.apply_displacement(model,last_disp)
            reuse_next=True; fresh+=1; event='fresh_adam'; batch_loss=f0
        rows.append({'algorithm':'periodic_k2_noquery','update':u,'event':event,'batch_loss':batch_loss,
                     'full_forward_passes':ff,'probe_forward_passes':pf,'backward_passes':bp,
                     'calibrated_nn_compute_seconds':ff*F+pf*Fp+bp*B})
    v081.sync(device); wall=time.perf_counter()-w0
    return rows,evaluate_summary(model,xtr,ytr,xev,yev,loss_fn,eval_batch_size,'periodic_k2_noquery',updates,
                                 ff,pf,bp,ffs,pfs,bs,wall,F,Fp,B,exf,exb,updates//2,accepts,0,fresh)


def run_microprobe_fggr(model, init_state, xtr, ytr, xev, yev, loss_fn, device, updates, batch_size, probe_size,
                        seed, lr, F, Fp, B, eval_batch_size):
    model.load_state_dict(init_state); model.to(device); model.train()
    state=v4g.init_adam_state(model)
    stream=v012.make_batch_stream(int(xtr.shape[0]),batch_size,seed)
    rows=[]; ff=pf=bp=0; ffs=pfs=bs=0.0; exf=exb=0; adam_t=0
    last_disp: Optional[Dict[str,torch.Tensor]]=None; eligible=False
    attempts=accepts=rejects=fresh=0
    v081.sync(device); w0=time.perf_counter()
    for u in range(1,updates+1):
        idx=next(stream)
        if eligible and last_disp is not None:
            pidx=idx[:probe_size]; xp,yp=v012.batch_from_indices(xtr,ytr,pidx,device)
            f0,dt0=timed_probe_forward(model,xp,yp,loss_fn,device)
            # Efficient live-parameter trial. If accepted, keep the displacement;
            # if rejected, restore before the full-gradient query.
            add_displacement_(model,last_disp,alpha=1.0)
            f1,dt1=timed_probe_forward(model,xp,yp,loss_fn,device)
            pf+=2; pfs+=dt0+dt1; exf+=2*int(xp.shape[0]); attempts+=1
            if np.isfinite(f1) and f1 < f0:
                accepts+=1; eligible=False
                event='accept_stale_microprobe_skip_full_gradient'; used=f1
            else:
                add_displacement_(model,last_disp,alpha=-1.0)
                xb,yb=v012.batch_from_indices(xtr,ytr,idx,device)
                loss,fb,dtf=v4g.timed_forward_graph(model,xb,yb,loss_fn,device); dtb=v4g.timed_backward(loss,device)
                ff+=1; bp+=1; ffs+=dtf; bs+=dtb; exf+=int(xb.shape[0]); exb+=int(xb.shape[0]); adam_t+=1
                last_disp=v4g.adam_displacement_from_current_grads(model,state,adam_t,lr); v4g.apply_displacement(model,last_disp)
                rejects+=1; eligible=True; event='reject_microprobe_pay_full_gradient'; used=fb
        else:
            xb,yb=v012.batch_from_indices(xtr,ytr,idx,device)
            loss,fb,dtf=v4g.timed_forward_graph(model,xb,yb,loss_fn,device); dtb=v4g.timed_backward(loss,device)
            ff+=1; bp+=1; ffs+=dtf; bs+=dtb; exf+=int(xb.shape[0]); exb+=int(xb.shape[0]); adam_t+=1
            last_disp=v4g.adam_displacement_from_current_grads(model,state,adam_t,lr); v4g.apply_displacement(model,last_disp)
            eligible=True; fresh+=1; event='forced_fresh_adam'; used=fb
        rows.append({'algorithm':f'fggr_microprobe{probe_size}','update':u,'event':event,'batch_loss':used,
                     'full_forward_passes':ff,'probe_forward_passes':pf,'backward_passes':bp,
                     'reuse_attempts':attempts,'reuse_accepts':accepts,'reuse_rejects':rejects,
                     'reuse_accept_rate':accepts/max(attempts,1),
                     'calibrated_nn_compute_seconds':ff*F+pf*Fp+bp*B})
    v081.sync(device); wall=time.perf_counter()-w0
    return rows,evaluate_summary(model,xtr,ytr,xev,yev,loss_fn,eval_batch_size,f'fggr_microprobe{probe_size}',updates,
                                 ff,pf,bp,ffs,pfs,bs,wall,F,Fp,B,exf,exb,attempts,accepts,rejects,fresh)


def write_plots(summaries: List[dict], out: Path) -> None:
    labels=[s['algorithm'] for s in summaries]; acc=[100*s['final_eval_accuracy'] for s in summaries]
    for key,xlabel,fname in [
        ('backward_passes','Backward passes','accuracy_vs_backward_passes.png'),
        ('calibrated_nn_compute_seconds','Calibrated pure NN compute (s)','accuracy_vs_calibrated_compute.png'),
        ('wall_seconds','End-to-end training wall time (s)','accuracy_vs_wall_time.png')]:
        fig,ax=plt.subplots(figsize=(9,6))
        for s,y in zip(summaries,acc): ax.scatter(s[key],y,s=90,label=s['algorithm'])
        ax.set_xlabel(xlabel); ax.set_ylabel('Final CIFAR-10 eval accuracy (%)')
        ax.set_title('Stage 5A1 CIFAR-10 ResNet-18-GN development')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/fname,dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['cifar10','fake'],default='cifar10')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--train-subset',type=int,default=50000); p.add_argument('--eval-subset',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=256); p.add_argument('--probe-size',type=int,default=128)
    p.add_argument('--updates',type=int,default=1500); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda'); p.add_argument('--adam-lr',type=float,default=0.001)
    p.add_argument('--eval-batch-size',type=int,default=512)
    p.add_argument('--out',default='stage5a1_cifar10_resnet18gn_fggr_probe128_seed0')
    p.add_argument('--algorithms',nargs='+',default=['adam_tensorwise','fggr_microprobe','periodic_k2_noquery'])
    args=p.parse_args()
    if args.probe_size <= 0 or args.probe_size > args.batch_size: raise ValueError('probe-size must be in [1,batch-size]')
    device=torch.device('cuda' if args.device=='cuda' and torch.cuda.is_available() else args.device)
    v081.set_seed(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.dataset=='cifar10':
        xtr,ytr,xev,yev=load_cifar10_tensors(Path(args.data_dir),args.train_subset,args.eval_subset,args.seed,device)
    else:
        xtr,ytr,xev,yev=load_fake_cifar(args.train_subset,args.eval_subset,args.seed,device)
    loss_fn=nn.CrossEntropyLoss(); model=make_cifar_resnet18_gn().to(device); init_state=v4g.clone_state_dict(model)
    n_params=v012.parameter_count(model)

    stream=v012.make_batch_stream(int(xtr.shape[0]),args.batch_size,args.seed); idx=next(stream)
    xb,yb=v012.batch_from_indices(xtr,ytr,idx,device); pidx=idx[:args.probe_size]; xp,yp=v012.batch_from_indices(xtr,ytr,pidx,device)
    fullcal=v012.calibrate_pure_batch_costs(model,init_state,xb,yb,loss_fn,device)
    probecal=v012.calibrate_pure_batch_costs(model,init_state,xp,yp,loss_fn,device)
    F=float(fullcal['pure_forward_median_s']); B=float(fullcal['pure_backward_median_s']); Fp=float(probecal['pure_forward_median_s'])
    pbe=2*Fp/max(F+B,1e-15)

    print(f'Device={device}; GPU={torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"}')
    print(f'Dataset={args.dataset}; model=CIFARResNet18GN; params={n_params:,}; batch={args.batch_size}; probe={args.probe_size}')
    print(f'Calib full F={F:.6f}s B={B:.6f}s; probe F={Fp:.6f}s; break-even acceptance={pbe:.3f}')
    print('Execution order:',args.algorithms)

    summaries=[]; traces={}
    for alg in args.algorithms:
        print('\nRUNNING',alg,flush=True)
        common=(model,init_state,xtr,ytr,xev,yev,loss_fn,device,args.updates,args.batch_size,args.seed,args.adam_lr,F,Fp,B,args.eval_batch_size)
        if alg=='adam_tensorwise': rows,s=run_adam(*common)
        elif alg=='periodic_k2_noquery': rows,s=run_periodic_noquery(*common)
        elif alg=='fggr_microprobe':
            rows,s=run_microprobe_fggr(model,init_state,xtr,ytr,xev,yev,loss_fn,device,args.updates,args.batch_size,args.probe_size,args.seed,args.adam_lr,F,Fp,B,args.eval_batch_size)
        else: raise ValueError(alg)
        s.update({'seed':args.seed,'dataset':args.dataset,'model':'CIFARResNet18GN','model_parameters':n_params,
                  'train_subset':int(xtr.shape[0]),'eval_subset':int(xev.shape[0]),'batch_size':args.batch_size,'probe_size':args.probe_size,
                  'adam_lr':args.adam_lr,'calib_full_forward_median_s':F,'calib_backward_median_s':B,'calib_probe_forward_median_s':Fp,
                  'microprobe_break_even_accept_rate':pbe,'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                  'torch_version':torch.__version__,'python_version':platform.python_version(),'execution_order':' > '.join(args.algorithms)})
        summaries.append(s); traces[s['algorithm']]=rows; v081.write_csv(out/f'trace_{s["algorithm"]}.csv',rows)

    v081.write_csv(out/'summary.csv',summaries)
    by={s['algorithm']:s for s in summaries}; a=by['adam_tensorwise']; paired=[]
    for s in summaries:
        if s['algorithm']=='adam_tensorwise': continue
        paired.append({'seed':args.seed,'algorithm':s['algorithm'],'execution_order':s['execution_order'],
                       'adam_eval_accuracy':a['final_eval_accuracy'],'method_eval_accuracy':s['final_eval_accuracy'],
                       'accuracy_delta_pp':100*(s['final_eval_accuracy']-a['final_eval_accuracy']),
                       'adam_backward_passes':a['backward_passes'],'method_backward_passes':s['backward_passes'],
                       'backward_reduction_pct':100*(a['backward_passes']-s['backward_passes'])/max(a['backward_passes'],1),
                       'adam_calibrated_nn_seconds':a['calibrated_nn_compute_seconds'],'method_calibrated_nn_seconds':s['calibrated_nn_compute_seconds'],
                       'calibrated_nn_reduction_pct':100*(a['calibrated_nn_compute_seconds']-s['calibrated_nn_compute_seconds'])/max(a['calibrated_nn_compute_seconds'],1e-15),
                       'adam_direct_nn_seconds':a['nn_compute_seconds'],'method_direct_nn_seconds':s['nn_compute_seconds'],
                       'direct_nn_time_reduction_pct':100*(a['nn_compute_seconds']-s['nn_compute_seconds'])/max(a['nn_compute_seconds'],1e-15),
                       'adam_wall_seconds':a['wall_seconds'],'method_wall_seconds':s['wall_seconds'],
                       'wall_time_reduction_pct':100*(a['wall_seconds']-s['wall_seconds'])/max(a['wall_seconds'],1e-15),
                       'reuse_attempts':s['reuse_attempts'],'reuse_accepts':s['reuse_accepts'],'reuse_rejects':s['reuse_rejects'],
                       'reuse_accept_rate':s['reuse_accept_rate'],'microprobe_break_even_accept_rate':pbe,
                       'accept_rate_margin':s['reuse_accept_rate']-pbe})
    v081.write_csv(out/'paired_vs_adam.csv',paired)

    if 'fggr_microprobe128' in by and 'periodic_k2_noquery' in by:
        f=by['fggr_microprobe128']; q=by['periodic_k2_noquery']
        v081.write_csv(out/'fggr_vs_periodic_head_to_head.csv',[{
            'seed':args.seed,
            'fggr_accuracy':f['final_eval_accuracy'],'periodic_accuracy':q['final_eval_accuracy'],
            'fggr_minus_periodic_accuracy_pp':100*(f['final_eval_accuracy']-q['final_eval_accuracy']),
            'fggr_backward_passes':f['backward_passes'],'periodic_backward_passes':q['backward_passes'],
            'fggr_calibrated_nn_seconds':f['calibrated_nn_compute_seconds'],'periodic_calibrated_nn_seconds':q['calibrated_nn_compute_seconds'],
            'fggr_wall_seconds':f['wall_seconds'],'periodic_wall_seconds':q['wall_seconds'],
            'fggr_reuse_accept_rate':f['reuse_accept_rate'],'break_even_accept_rate':pbe,
        }])

    write_plots(summaries,out)
    report=[
        'AGRLS Stage 5A1 - CIFAR-10 ResNet-18-GN Forward-Gated Gradient Refresh development\n',
        f'device={device}; gpu={torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"}\n',
        f'parameters={n_params:,}; batch={args.batch_size}; probe={args.probe_size}; updates={args.updates}; seed={args.seed}\n',
        f'calibration full_F={F:.6f}s full_B={B:.6f}s probe_F={Fp:.6f}s pBE={pbe:.4f}\n\n'
    ]
    for s in summaries:
        report.append(f"{s['algorithm']:22s} eval={100*s['final_eval_accuracy']:.2f}% bwd={s['backward_passes']} "
                      f"calib={s['calibrated_nn_compute_seconds']:.3f}s directNN={s['nn_compute_seconds']:.3f}s "
                      f"wall={s['wall_seconds']:.3f}s reuse_accept={s['reuse_accept_rate']:.3f}\n")
    (out/'REPORT.txt').write_text(''.join(report),encoding='utf-8')
    print('\n'+''.join(report))


if __name__=='__main__':
    main()
