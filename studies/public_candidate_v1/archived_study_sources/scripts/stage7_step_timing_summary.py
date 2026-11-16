"""Aggregate recorded step timings by full and partial update modes."""
from __future__ import annotations
import argparse,csv,statistics
from pathlib import Path

def percentile(xs,p):
    ys=sorted(xs)
    if not ys:return None
    pos=(len(ys)-1)*p
    a=int(pos);b=min(a+1,len(ys)-1)
    return ys[a]+(ys[b]-ys[a])*(pos-a)

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--root',required=True)
    args=pa.parse_args(); root=Path(args.root)
    rows=[]
    for seed_dir in sorted(root.glob('seed_*')):
        for tr in sorted(seed_dir.glob('*/step_trace.csv')):
            if not (tr.parent/'summary.json').exists():
                continue
            raw=list(csv.DictReader(open(tr,encoding='utf8')))
            if not raw:continue
            for mode in (0,1):
                subset=[r for r in raw if int(r['partial'])==mode]
                if not subset:continue
                out={'seed':int(seed_dir.name.rsplit('_',1)[1]),'variant':tr.parent.name,
                    'step_type':'partial' if mode else 'full','count':len(subset)}
                for key in ('aug_ms','forward_ms','backward_ms','optimizer_ms'):
                    vs=[float(r[key]) for r in subset]
                    out[key+'_mean']=statistics.mean(vs)
                    out[key+'_median']=statistics.median(vs)
                    out[key+'_p10']=percentile(vs,.1)
                    out[key+'_p90']=percentile(vs,.9)
                rows.append(out)
    if not rows: raise FileNotFoundError('No step_trace.csv files found. Use --log-steps when running the trainer.')
    path=root/'stage7_step_mode_timing_summary.csv'
    with open(path,'w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Wrote',path,'from',len(rows),'seed x variant x step-type groups')

if __name__=='__main__':main()
