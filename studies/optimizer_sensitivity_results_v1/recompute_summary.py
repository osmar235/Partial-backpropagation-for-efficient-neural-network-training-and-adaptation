"""Portable summary reconstruction from exported comparisons (not raw tensors)."""
import json
from pathlib import Path
from statistics import median

def recompute(path):
    rows=json.loads(Path(path).read_text(encoding='utf8'));assert len(rows)==132
    bf16=[r for r in rows if 'bf16' in r['arm']];assert len(bf16)==88
    groups=[]
    for key in sorted({(r['kind'],r['domain'],r['anchor'],r['context']) for r in bf16}):
        group=[r for r in bf16 if (r['kind'],r['domain'],r['anchor'],r['context'])==key]
        metrics=['gradient_relative_l2','update_relative_l2','max_update_difference','sign_flip_fraction','small_gradient_mass_lt_1e6','small_gradient_mass_lt_1e4','exploratory_key_bias_mass','top32_max_analytic_error']
        groups.append(dict(kind=key[0],domain=key[1],anchor=key[2],context=key[3],comparisons=len(group),
            ranges={k:dict(min=min(r[k] for r in group),max=max(r[k] for r in group),median=median(r[k] for r in group)) for k in metrics}))
    return groups

if __name__=='__main__':
    print(json.dumps(recompute(Path(__file__).resolve().parent/'comparison_rows.json'),indent=2))
