"""Read-only integrity/arithmetic audit of growth feasibility, not efficacy."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
import statistics as st
from pathlib import Path
import torch
import numpy as np
import stage10_models as m
import stage10_representation as old
import stage11_growth as g
import stage11_growth_feasibility as run
import stage9_boundary_diagnostic as d
import stage8_vit_v2 as v


def audit(out,write=False,attempt=None):
    m.configure(); out = Path(out)
    cfg = json.loads((out/'configuration.json').read_text()); spec = cfg['spec']
    assert v.digest({k:x for k,x in cfg.items() if k!='fingerprint'}) == cfg['fingerprint']
    freeze = out/'SYNTHETIC_FREEZE.json' if cfg['synthetic'] else run.FREEZE_PATH
    record = json.loads(freeze.read_text())
    assert record == run.freeze_record(spec,cfg['synthetic'])
    assert record['freeze_hash'] == cfg['freeze_hash']
    assert cfg['spec'] == record['spec'] and cfg['source_sha256'] == record['source_sha256']
    assert cfg['split'] == record['expected_split'] and cfg['checkpoints'] == record['checkpoints']
    assert cfg['execution'] == record['required_execution'] == m.execution_settings()
    if (out/'COMPLETE.json').exists():
        assert not write
        marker = d.validate_complete(out,cfg['fingerprint']); attempt = Path(marker['attempt'])
    else:
        assert write and attempt is not None
        attempt = Path(attempt)
    states = json.loads((attempt/'input_states.json').read_text())
    live = json.loads((attempt/'liveness_summary.json').read_text())
    indices = json.loads((attempt/'probe_indices.json').read_text())
    eq = d.records(attempt/'equivalence.csv'); rows = d.records(attempt/'measurements.csv')
    live_rows = d.records(attempt/'liveness.csv'); cold = d.records(attempt/'cold_steps.csv')
    nval = 8 if cfg['synthetic'] else 5000
    ntrain = 16 if cfg['synthetic'] else 45000
    assert len(eq) == nval * len(spec['source_seeds'])
    assert len(rows) == spec['probe_batches'] * 3 * len(spec['source_seeds'])
    assert len(live_rows) == spec['liveness_steps'] * 3 * len(spec['source_seeds'])
    assert len(states) == len(indices) == len(spec['source_seeds'])
    assert len(live) == len(cold) == 3 * len(spec['source_seeds'])
    summaries = []
    for seed in spec['source_seeds']:
        state = next(r for r in states if r['seed']==seed)
        assert math.isfinite(state['growth_seconds']) and state['growth_seconds'] > 0
        assert state['parameter_counts']['grown'] == state['parameter_counts']['full']
        assert state['parameter_counts']['small'] < state['parameter_counts']['full']
        ix = next(r for r in indices if r['seed']==seed)['indices']
        gen = torch.Generator().manual_seed(seed+220001)
        expected_ix = [torch.randperm(ntrain,generator=gen)[:spec['batch_size']].tolist() for _ in range(spec['probe_batches'])]
        assert ix == expected_ix
        es = [r for r in eq if int(r['seed'])==seed]
        assert [int(r['index']) for r in es] == list(range(nval))
        for r in es:
            assert all(0 <= int(r[k]) <= 9 for k in ['label','small_prediction','grown_prediction'])
            assert all(math.isfinite(float(r[k])) and float(r[k]) >= 0 for k in
                       ['small_loss','grown_loss','max_abs_logit_error','max_scaled_logit_error'])
        small_acc = st.mean(int(r['small_prediction'])==int(r['label']) for r in es)
        grown_acc = st.mean(int(r['grown_prediction'])==int(r['label']) for r in es)
        changes = sum(r['small_prediction']!=r['grown_prediction'] for r in es)
        max_scaled = max(float(r['max_scaled_logit_error']) for r in es)
        preservation = changes <= spec['maximum_prediction_changes'] and max_scaled <= 1.
        if not cfg['synthetic']:
            for variant,path in [('small_rgb32','small'),('full_rgb32','full')]:
                entry = record['checkpoints'][f'{seed}:{variant}']
                cp = torch.load(entry['path'],map_location='cpu',weights_only=True)
                model = m.make_model(variant,seed)
                model.load_state_dict(cp['model_state'])
                assert old.state_hash(model.state_dict()) == state['initial_state_sha256'][path]
                assert sum(p.numel() for p in model.parameters()) == state['parameter_counts'][path]
                if path == 'small':
                    expanded = g.expand(model,seed)
                    assert old.state_hash(expanded.state_dict()) == state['initial_state_sha256']['grown']
                    source = json.loads(Path(entry['summary_path']).read_text())
                    assert abs(small_acc-source['final_validation_accuracy']) < 1e-12
                    assert abs(st.mean(float(r['small_loss']) for r in es)-source['final_validation_loss']) < 1e-6
        seconds = {}
        for path in spec['paths']:
            ls = next(r for r in live if r['seed']==seed and r['path']==path)
            assert ls['finite'] and ls['optimizer_reset']
            assert ls['minimum_step'] == ls['maximum_step'] == spec['liveness_steps']
            assert ls['state_before'] == state['initial_state_sha256'][path] != ls['state_after']
            lr = [r for r in live_rows if int(r['seed'])==seed and r['path']==path]
            assert [int(r['step']) for r in lr] == list(range(1,spec['liveness_steps']+1))
            assert all(math.isfinite(float(r['loss'])) and float(r['loss']) >= 0 for r in lr)
            cp = torch.load(attempt/f'liveness_{seed}_{path}.pt',map_location='cpu',weights_only=True)
            assert cp['seed']==seed and cp['path']==path and cp['steps']==spec['liveness_steps']
            assert old.state_hash(cp['model_state']) == ls['state_after']
            model = m.make_model('small_rgb32' if path=='small' else 'full_rgb32',seed,cfg['synthetic'])
            model.load_state_dict(cp['model_state'],strict=True)
            ids = [i for group in cp['optimizer_state']['param_groups'] for i in group['params']]
            assert len(ids)==len(cp['optimizer_state']['state'])==len(list(model.parameters()))
            for param,i in zip(model.parameters(),ids):
                opt = cp['optimizer_state']['state'][i]
                assert int(opt['step'])==spec['liveness_steps'] and torch.isfinite(param).all()
                for name in ['exp_avg','exp_avg_sq']:
                    assert opt[name].shape==param.shape and torch.isfinite(opt[name]).all()
            if path=='grown':
                assert g.twin_difference(model)==ls['grown_embedding_twin_difference'] > 1e-8
            data = [r for r in rows if int(r['seed'])==seed and r['path']==path]
            assert [int(r['probe']) for r in data] == list(range(spec['probe_batches']))
            assert all(math.isfinite(float(r['seconds'])) and float(r['seconds'])>0 for r in data)
            for i,r in enumerate(data):
                order = spec['paths'][i%3:]+spec['paths'][:i%3]
                if (i//3)%2:order=list(reversed(order))
                assert int(r['order_position'])==order.index(path)
                assert int(r['batch_examples'])==spec['batch_size']
                assert int(r['dropout_seed'])==seed+224001+i
                lam = np.random.default_rng(seed+222001+i).beta(spec['mixup_alpha'],spec['mixup_alpha'])
                assert abs(float(r['mixup_lambda'])-lam)<1e-12
                assert math.isfinite(float(r['loss'])) and float(r['loss'])>=0
            seconds[path]=st.mean(float(r['seconds']) for r in data)
            c = next(r for r in cold if int(r['seed'])==seed and r['path']==path)
            assert math.isfinite(float(c['seconds'])) and float(c['seconds'])>0
        small_ratio = seconds['small']/seconds['full']; grown_ratio = seconds['grown']/seconds['full']
        cost = small_ratio <= spec['maximum_small_full_step_ratio'] and grown_ratio <= spec['maximum_grown_full_step_ratio']
        summaries.append(dict(seed=seed,small_accuracy=small_acc,grown_accuracy=grown_acc,prediction_changes=changes,
                              max_abs_logit_error=max(float(r['max_abs_logit_error']) for r in es),max_scaled_logit_error=max_scaled,
                              preservation_pass=preservation,growth_seconds=state['growth_seconds'],
                              small_seconds=seconds['small'],grown_seconds=seconds['grown'],full_seconds=seconds['full'],
                              small_full_step_ratio=small_ratio,grown_full_step_ratio=grown_ratio,cost_headroom_pass=cost,
                              liveness_pass=True,feasible_for_pilot_review=preservation and cost))
    result = dict(status='PASS',synthetic=cfg['synthetic'],fingerprint=cfg['fingerprint'],
                  source_states=len(spec['source_seeds'])*2,measured_operation_rows=len(rows),
                  validation_examples_checked=len(eq),liveness_updates=len(live_rows),
                  all_sources_and_checkpoints_verified=True,all_disposable_parameters_updated=True,
                  both_seed_preservation_pass=all(r['preservation_pass'] for r in summaries),
                  both_seed_cost_headroom_pass=all(r['cost_headroom_pass'] for r in summaries),
                  feasible_for_pilot_review=all(r['feasible_for_pilot_review'] for r in summaries),
                  inference=spec['inference'])
    if write:
        v.write_csv(attempt/'summary.csv',summaries); v.atomic_json(attempt/'audit.json',result)
        lines=['# Stage11A growth feasibility','',spec['inference'],'',
               '| Seed | Small accuracy | Grown accuracy | Prediction changes | Small/full step time | Grown/full step time | Feasible for pilot review |',
               '|---:|---:|---:|---:|---:|---:|---|']
        for r in summaries:
            lines.append(f'| {r["seed"]} | {100*r["small_accuracy"]:.2f}% | {100*r["grown_accuracy"]:.2f}% | {r["prediction_changes"]} | {r["small_full_step_ratio"]:.4f} | {r["grown_full_step_ratio"]:.4f} | {r["feasible_for_pilot_review"]} |')
        lines += ['',f'Synthetic: {cfg["synthetic"]}. Audit PASS describes integrity, not efficacy.',
                  'Warm operation times include augmentation, MixUp, seeding, forward/backward and AdamW.',
                  'Growth construction and cold optimizer-allocation step times are separately recorded.',
                  'All benchmark steps restore source weights and fresh zero Adam moments outside the timer.',
                  'Disposable short updates check trainability, not convergence or accuracy recovery.',
                  'Dropout-free function preservation does not preserve stochastic dropout or Adam trajectories.',
                  'This must not be interpreted as observed time-to-quality or a novel training method.',
                  'Review before choosing a single prospective growth pilot; no automatic launch.','']
        (attempt/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    else:
        assert json.loads((attempt/'audit.json').read_text())==result
        original=d.records(attempt/'summary.csv')
        assert len(original)==len(summaries)
        for actual,saved in zip(summaries,original):
            for k,value in actual.items():
                if isinstance(value,bool):assert str(value)==saved[k]
                else:assert math.isclose(float(value),float(saved[k]),rel_tol=1e-10,abs_tol=1e-12)
    print(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=run.REAL_OUT)
    audit(p.parse_args().root,write=False)
