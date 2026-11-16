"""Bounded compiler capability probe; no package installation or full study."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]

def child(path):
    import torch
    import torch._dynamo
    record=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
        system=platform.system(),triton_installed=importlib.util.find_spec('triton') is not None,
        cuda_available=torch.cuda.is_available(),dynamo_supported=torch._dynamo.is_dynamo_supported())
    if torch.cuda.is_available():
        record.update(hardware=torch.cuda.get_device_name(0),bf16_supported=torch.cuda.is_bf16_supported())
    torch._dynamo.config.suppress_errors=False
    start=time.perf_counter()
    try:
        def fn(x):return (x.sin()*x).sum()
        compiled=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False)
        x=torch.ones(16,device='cuda',requires_grad=True)
        y=compiled(x);y.backward();torch.cuda.synchronize()
        record.update(compile_status='SUPPORTED_TINY_PROBE_ONLY',output=float(y),gradient_finite=bool(torch.isfinite(x.grad).all()))
    except Exception as error:
        record.update(compile_status='UNAVAILABLE_IN_PROBE',exception_type=type(error).__name__,reason=str(error))
    record['compile_first_call_seconds']=time.perf_counter()-start
    path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--child',type=Path);args=ap.parse_args()
    if args.child:child(args.child);return
    out=ROOT/'results/optimized_execution_preparation_v1';out.mkdir(exist_ok=True)
    result=out/'environment_probe.json';assert not result.exists()
    env=dict(os.environ);env['TORCHINDUCTOR_CACHE_DIR']=str(out/'compiler_cache')
    env['TRITON_CACHE_DIR']=str(out/'triton_cache')
    with (out/'environment_probe.log').open('w',encoding='utf8') as log:
        try:
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child',str(result)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        except subprocess.TimeoutExpired:
            result.write_text(json.dumps(dict(compile_status='PROBE_TIMEOUT',timeout_seconds=120))+'\n');raise
    assert proc.returncode==0 and result.exists(),(out/'environment_probe.log').read_text()
    print(result.read_text())

if __name__=='__main__':main()
