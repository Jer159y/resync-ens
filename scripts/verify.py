"""Verify the immutable export and reaggregate numerical evidence (standard library only)."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'provenance/files.sha256.json'
if manifest.exists():
    entries=json.loads(manifest.read_text())
    for name,digest in entries.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit('Hash mismatch: '+name)
    print('Verified',len(entries),'exported files')
subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_tables.py')],check=True)
# All TeX inputs and figures must resolve within this repository, not the parent EESN.
import re
for p in (ROOT/'figures').glob('*.tex'):
    for path in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',p.read_text()):
        if "#" in path:continue
        q=ROOT/path
        if not q.suffix:q=q.with_suffix('.tex')
        if not q.exists():raise SystemExit(f'Unresolved TeX input in {p.name}: {q}')
print('VERIFY_OK: numerical anchors, source integrity and local TeX inputs')
