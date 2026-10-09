"""Standard-library provenance, statistics and stage records."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def json_write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.new')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(p)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def finite(value):
    try: return math.isfinite(float(value))
    except (ValueError, TypeError): return False

def summary(values):
    x=[float(v) for v in values]
    if not x or not all(math.isfinite(v) for v in x): raise ValueError('Missing/nonfinite required measurements')
    med=statistics.median(x)
    return {'n':len(x),'mean':statistics.mean(x),'sample_sd':statistics.stdev(x) if len(x)>1 else None,
            'median':med,'mad':statistics.median(abs(v-med) for v in x)}

def expression(text):
    """Arithmetic only. Config validation must not execute arbitrary INI code."""
    tree=ast.parse(str(text), mode='eval')
    allowed=(ast.Expression,ast.Constant,ast.UnaryOp,ast.UAdd,ast.USub,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Mod)
    if any(not isinstance(n,allowed) for n in ast.walk(tree)): raise ValueError('Not a numeric expression: '+str(text))
    return eval(compile(tree,'<arithmetic>','eval'),{'__builtins__':{}},{})

def fingerprints(paths, root):
    root=Path(root).resolve(); out={}
    for p in paths:
        p=Path(p).resolve(); rel=p.relative_to(root).as_posix()
        out[rel]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    return out

def validate_files(record, root):
    for rel, pin in record.items():
        p=Path(root)/rel
        if not p.is_file() or p.stat().st_size!=pin['bytes'] or sha256(p)!=pin['sha256']:
            raise ValueError('Artifact identity mismatch: '+str(p))

def package_digest():
    lock=ROOT/'PACKAGE_FILES.json'
    pins=json_read(lock)
    validate_files(pins,ROOT)
    return sha256(lock)
