"""Path-normalizing, read-only entry point for the frozen backend auditor.

The frozen auditor assumes an absolute output path. Preserve its source hash
and normalize the CLI path here; never launch or repeat the experiment.
"""
import argparse
import json
from pathlib import Path
import efficient_backprop as experiment
from review_efficient_backprop import audit

def review(output):
    output=Path(output).resolve(strict=True)
    cfg=json.loads((output/'configuration.json').read_text(encoding='utf8'))
    assert experiment.d.validate_complete(output,cfg['fingerprint']),'Completed root marker required'
    return audit(output,False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path,nargs='?',default=experiment.OUT)
    args=parser.parse_args();experiment.m.configure()
    print(json.dumps(review(args.output),indent=2))
