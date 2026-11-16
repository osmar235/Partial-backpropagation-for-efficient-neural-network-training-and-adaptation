"""Verify the exact distributed files; ignore Git metadata and replay outputs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def included(p):
    rel = p.relative_to(ROOT)
    return p.is_file() and not any(x in {'.git', '.venv', '__pycache__', 'reproduced'} for x in rel.parts) and rel.as_posix() != 'SHA256SUMS.json'

def verify():
    expected = json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf8'))
    actual = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*') if included(p)}
    if set(actual) != set(expected):
        raise RuntimeError(dict(missing=sorted(set(expected)-set(actual)), extra=sorted(set(actual)-set(expected))))
    for name, path in actual.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise RuntimeError('Hash mismatch: '+name)
    return dict(status='PASS', verified_files=len(expected))

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
