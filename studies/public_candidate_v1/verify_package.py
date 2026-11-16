"""Verify every packaged file, including unexpected additions, without importing torch."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    expected = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf8'))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.name != 'SHA256SUMS.json'
              and '__pycache__' not in p.parts}
    assert actual == set(expected), {'missing': sorted(set(expected)-actual), 'extra': sorted(actual-set(expected))}
    for name, digest in expected.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    return {'status': 'PASS', 'verified_files': len(expected)}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
