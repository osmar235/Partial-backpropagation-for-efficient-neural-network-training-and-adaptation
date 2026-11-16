"""Read-only raw-trace review; no torch import or optimization updates.

Correlate device events to host launch records, then to CPU operators. Assign
phases by enclosing host ranges across threads. Do not assign a kernel by the
GPU timestamp: asynchronous kernels can execute after the host phase ends.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/runtime_profile_v1';OUT=ROOT/'results/runtime_profile_review_v1'
PHASES=('seed_zero','forward','loss_readback','backward','optimizer')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def rows(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
def table(path,data):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def verify(marker):
    for p,h in marker['sha256'].items():assert sha(p)==h,p

def trace_analysis(path,seed,batch,arm):
    events=read(path)['traceEvents']
    ranges=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('phase::')]
    modules=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('module::')]
    for phase in PHASES:assert sum(e['name']=='phase::'+phase for e in ranges)==3
    ordered=sorted(ranges,key=lambda e:e['ts'])
    assert all(a['ts']+a['dur']<=b['ts']+1 for a,b in zip(ordered,ordered[1:]))
    def phase_of(event):
        # Host timestamps share the same process clock across autograd threads.
        matches=[p for p in ranges if p['pid']==event['pid'] and p['ts']-1<=event['ts'] and event['ts']+event.get('dur',0)<=p['ts']+p['dur']+1]
        assert len(matches)<=1
        return matches[0] if matches else None
    cpus={};launches={}
    for e in events:
        args=e.get('args',{})
        if e.get('cat') in ('cpu_op','user_annotation') and 'External id' in args:
            key=(e['pid'],args['External id']);assert key not in cpus;cpus[key]=e
        if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in args:
            launches.setdefault(args['correlation'],[]).append(e)
    groups={};unmatched=collections.Counter();unmatched_us=collections.Counter();cross_thread=0
    for e in events:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        matches=launches.get(e.get('args',{}).get('correlation'),[])
        if not matches:
            reason='no_captured_launch';unmatched[reason]+=1;unmatched_us[reason]+=e['dur'];continue
        # Runtime and driver records can nest; prefer the linked CPU operator.
        linked=[]
        for launch in matches:
            op=cpus.get((launch['pid'],launch.get('args',{}).get('External id')))
            phase=phase_of(op or launch)
            if phase is not None:linked.append((op,launch,phase))
        if not linked:
            reason='launch_outside_active_phases';unmatched[reason]+=1;unmatched_us[reason]+=e['dur'];continue
        assert len({item[2]['name'] for item in linked})==1
        op,launch,phase=next((z for z in linked if z[0] is not None),linked[0])
        if op is not None and op['tid']!=phase['tid']:cross_thread+=1
        shapes=(op or {}).get('args',{}).get('Input Dims',[])
        module=''
        if phase['name']=='phase::forward' and op is not None:
            candidates=[z for z in modules if z['pid']==op['pid'] and z['tid']==op['tid'] and z['ts']-1<=op['ts'] and op['ts']+op['dur']<=z['ts']+z['dur']+1]
            if candidates:module=min(candidates,key=lambda z:z['dur'])['name'][8:]
        key=(phase['name'][7:],module,(op or {}).get('name','UNLINKED_OPERATOR'),json.dumps(shapes,separators=(',',':')),e['cat'])
        row=groups.setdefault(key,dict(seed=seed,batch=batch,arm=arm,phase=key[0],forward_module=module,
            operator=key[2],input_shapes=key[3],device_category=e['cat'],device_events=0,device_us=0.))
        assert math.isfinite(e['dur']) and e['dur']>=0
        row['device_events']+=1;row['device_us']+=e['dur']
    grouped=list(groups.values());total=sum(z['device_us'] for z in grouped)
    dw=[];n=batch*65 if arm!='sampled_quarter' else batch*17
    expected=([768,n],[n,192]),([192,n],[n,768])
    for z in grouped:
        shapes=json.loads(z['input_shapes'])
        if z['phase']=='backward' and z['operator']=='aten::mm' and len(shapes)>=2 and tuple(shapes[:2]) in expected:dw.append(z)
    dw_kernels=[z for z in dw if z['device_category']=='kernel']
    assert len(dw_kernels)==2 and all(z['device_events']>=18 for z in dw_kernels),(seed,batch,arm,dw)
    for shape in expected:
        calls=[e for e in cpus.values() if e['name']=='aten::mm' and tuple(e.get('args',{}).get('Input Dims',[])[:2])==shape]
        assert len(calls)==18 and all(phase_of(e)['name']=='phase::backward' for e in calls)
    summary=dict(seed=seed,batch=batch,arm=arm,active_updates=3,assigned_device_events=sum(z['device_events'] for z in grouped),
        assigned_device_us=total,cross_thread_assigned_device_events=cross_thread,
        unassigned_device_events=sum(unmatched.values()),unassigned_device_us=sum(unmatched_us.values()),
        mlp_dw_device_ms_per_update=sum(z['device_us'] for z in dw)/3000,
        mlp_dw_percent_assigned_device=100*sum(z['device_us'] for z in dw)/total)
    for phase in PHASES:summary[phase+'_device_ms_per_update']=sum(z['device_us'] for z in grouped if z['phase']==phase)/3000
    write(OUT/f'trace_coverage_{seed}_{batch}_{arm}.json',dict(**summary,unassigned_reasons=dict(unmatched),
        unassigned_us_by_reason=dict(unmatched_us),method='GPU correlation -> host launch -> CPU external id -> enclosing host phase across threads',
        caveat='Summed device durations, not wall time; unassigned boundary/auxiliary events explicitly excluded.'))
    return summary,grouped

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'review_artifacts_sha256.json').exists()
    cfg=read(RAW/'configuration.json');freeze=read(ROOT/'protocol/runtime_profile/FROZEN_PROFILE_v1.json');assert cfg==freeze
    for p,h in cfg['sources'].items():assert sha(p)==h,p
    verify(read(RAW/'COMPLETE.json'))
    summaries=[];phases=[];trace_summaries=[];operators=[];count=0
    for parent in sorted(RAW.glob('seed_*/*')):
        rc=read(parent/'run_config.json');marker=read(parent/'COMPLETE.json');verify(marker)
        attempt=Path(marker['attempt']);assert attempt.parent==parent
        assert set(marker['sha256'])=={str(p) for p in attempt.rglob('*') if p.is_file()}
        seed,batch,arm=rc['seed'],rc['batch'],rc['arm'];base=dict(seed=seed,batch=batch,arm=arm)
        measurements={};identity=None;all_phases=[]
        for entry in read(attempt/'regimes.json'):
            folder=Path(entry['folder']);rec=read(folder/'measurement.json');mode=entry['mode'];ri=entry['round']
            assert rec['updates']==4 and rec['optimizer_age']==7
            assert rec['wall_seconds']==entry['wall_seconds'] and rec['wall_seconds']>0
            sig={k:rec[k] for k in ('losses','final_model_sha256','final_optimizer_sha256','cpu_rng_sha256','cuda_rng_sha256')}
            if identity is None:identity=sig
            assert sig==identity
            measurements[ri,mode]=rec;count+=4
            if mode=='events':all_phases+=rows(folder/'phases.csv')
        assert set(measurements)=={(ri,mode) for ri in range(3) for mode in ('plain','events')}|{(3,'profile')}
        plain=[measurements[ri,'plain']['wall_seconds']/4*1000 for ri in range(3)]
        event=[measurements[ri,'events']['wall_seconds']/4*1000 for ri in range(3)]
        summaries.append(dict(**base,plain_median_ms=st.median(plain),plain_min_ms=min(plain),plain_max_ms=max(plain),
            paired_event_overhead_percent=100*(st.median(b/a for a,b in zip(plain,event))-1),
            profile_session_to_plain_ratio=measurements[3,'profile']['wall_seconds']/(4*st.median(plain)/1000)))
        for phase in PHASES:
            values=[float(z['cuda_stream_ms']) for z in all_phases if z['phase']==phase];assert len(values)==12
            phases.append(dict(**base,phase=phase,event_median_ms=st.median(values),event_min_ms=min(values),event_max_ms=max(values)))
        trace_summary,grouped=trace_analysis(attempt/'03_profile/trace.json',seed,batch,arm)
        original_ops=rows(attempt/'03_profile/operators.csv')
        saved_totals=collections.defaultdict(float);rebuilt_totals=collections.defaultdict(float)
        for row in original_ops:saved_totals[row['operator']]+=float(row['self_device_us'])
        for row in grouped:rebuilt_totals[row['operator']]+=row['device_us']
        max_error=max(abs(saved_totals[k]-rebuilt_totals[k]) for k in set(saved_totals)|set(rebuilt_totals))
        assert max_error<1e-6,(base,max_error)
        trace_summary['max_operator_reconciliation_error_us']=max_error
        trace_summary['original_outside_device_us']=sum(float(z['self_device_us']) for z in original_ops if z['phase']=='outside')
        trace_summaries.append(trace_summary);operators+=grouped
    assert count==336 and len(summaries)==12
    comparisons=[]
    for seed in cfg['spec']['seeds']:
        for batch in cfg['spec']['batch_sizes']:
            subset={z['arm']:z for z in summaries if (z['seed'],z['batch'])==(seed,batch)}
            native=subset['native']['plain_median_ms'];sample=subset['sampled_quarter']['plain_median_ms']
            comparisons.append(dict(seed=seed,batch=batch,native_ms=native,sampled_ms=sample,
                sampled_slowdown_percent=100*(sample/native-1),exact_custom_ms=subset['exact_custom']['plain_median_ms']))
    table(OUT/'independent_timings.csv',summaries);table(OUT/'phase_event_timings.csv',phases)
    table(OUT/'implementation_comparisons.csv',comparisons);table(OUT/'trace_phase_attribution.csv',trace_summaries)
    table(OUT/'trace_operator_attribution.csv',operators)
    saved_summary={(int(z['seed']),int(z['batch']),z['arm']):z for z in rows(RAW/'summary.csv')}
    saved_phases={(int(z['seed']),int(z['batch']),z['arm'],z['phase']):z for z in rows(RAW/'phase_summary.csv')}
    for z in summaries:
        saved=saved_summary[z['seed'],z['batch'],z['arm']]
        assert abs(z['plain_median_ms']-float(saved['plain_median_update_ms']))<1e-8
        assert abs(z['paired_event_overhead_percent']-float(saved['median_paired_events_overhead_percent']))<1e-8
    for z in phases:assert abs(z['event_median_ms']-float(saved_phases[z['seed'],z['batch'],z['arm'],z['phase']]['median_cuda_stream_ms']))<1e-8
    write(OUT/'independent_audit.json',dict(status='PASS',conditions=12,measured_artificial_updates=count,
        burnin_artificial_updates=36,all_instrumented_updates_match=True,root_and_condition_hashes_verified=True,
        trace_cross_thread_phase_reconstruction=True,new_training_or_benchmark_performed=False,
        all_reconstructed_operator_totals_match_saved_profiler=True,all_timing_summaries_reproduced=True,
        interpretation='Original CPU-parent phase table omits most backward GPU work; use trace-correlation attribution plus phase events.'))
    print(json.dumps(dict(comparisons=comparisons,trace_summaries=trace_summaries),indent=2))

if __name__=='__main__':main()
