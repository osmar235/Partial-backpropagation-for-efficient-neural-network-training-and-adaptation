"""Portable timing recomputation from 192 saved trials; standard library only.

Does not establish numerical eligibility, provenance or real-data quality.
"""
import json
from pathlib import Path
from statistics import median

def recompute(path):
    rows=json.loads(Path(path).read_text(encoding='utf8'));assert len(rows)==192
    indexed={(r['session'],r['seed'],r['batch'],r['round'],r['arm']):r for r in rows}
    assert len(indexed)==192;result=[]
    for seed,batch in sorted({(r['seed'],r['batch']) for r in rows}):
        sessions=[]
        for session in range(4):
            ratios=[]
            for ri in range(6):
                native=indexed[session,seed,batch,ri,'native_fp32_math']['steady_seconds']
                reference=indexed[session,seed,batch,ri,'reference_fp32_math']['steady_seconds']
                assert native>0 and reference>0
                ratios.append(100*(native-reference)/native)
            sessions.append(median(ratios))
        result.append(dict(seed=seed,batch=batch,median_saving_percent=median(sessions),session_medians=sessions))
    return result

if __name__=='__main__':
    print(json.dumps(recompute(Path(__file__).resolve().parent/'trials_export.json'),indent=2))
