"""Maintainer utility: package a prepared inputs/ tree, recording per-file and archive hashes."""
from pathlib import Path
import sys,hashlib,json,tarfile
root=Path(sys.argv[1]).resolve()
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest={str(p.relative_to(root/'inputs')):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted((root/'inputs').rglob('*')) if p.is_file()}
(root/'inputs_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'main-inputs-v1.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(root/'inputs',arcname='inputs')
record={'name':archive.name,'sha256':digest(archive),'bytes':archive.stat().st_size,'n_files':len(manifest)}
(root/'artifact.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
