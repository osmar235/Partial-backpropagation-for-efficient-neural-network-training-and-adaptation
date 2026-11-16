"""Portable recomputation from the included 288 trial records; stdlib only.

This checks timing arithmetic, not source provenance, parameter tolerances or
raw checkpoint/trajectory correctness. See the accompanying independent review.
"""
import json
from pathlib import Path
from statistics import median

def recompute(path):
    rows=json.loads(Path(path).read_text(encoding='utf8'));assert len(rows)==288
    indexed={(r['session'],r['seed'],r['batch'],r['round'],r['arm']):r for r in rows}
    assert len(indexed)==288
    result=[]
    for seed,batch in sorted({(r['seed'],r['batch']) for r in rows}):
        for baseline,candidate in [('math','qkv_unbind'),('math','timm_reference'),('timm_reference','qkv_unbind')]:
            sessions=[]
            for session in range(4):
                ratios=[]
                for ri in range(6):
                    a=indexed[session,seed,batch,ri,baseline]['steady_seconds']
                    b=indexed[session,seed,batch,ri,candidate]['steady_seconds']
                    assert a>0 and b>0
                    ratios.append(100*(a-b)/a)
                sessions.append(median(ratios))
            result.append(dict(seed=seed,batch=batch,baseline=baseline,candidate=candidate,
                median_saving_percent=median(sessions),session_medians=sessions))
    return result

if __name__=='__main__':
    print(json.dumps(recompute(Path(__file__).resolve().parent/'trials_export.json'),indent=2))
