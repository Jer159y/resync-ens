"""Maintainer utility: update curated-source provenance and checksums after reviewed edits."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def files():
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if rel.parts[0] in {'.git','build','runs','outputs','artifacts'} or '__pycache__' in rel.parts:continue
        if rel.as_posix() in {'provenance/files.sha256.json','provenance/source_export.json'}:continue
        yield rel.as_posix(),p
items=dict(files());source=ROOT/'provenance/source_export.json';record=json.loads(source.read_text());seen=set()
for entry in record['files']:
    name=entry['destination'];seen.add(name)
    if name not in items:
        entry['status']='renamed_or_removed_from_final_export';continue
    digest=hashlib.sha256(items[name].read_bytes()).hexdigest()
    if entry.get('export_sha256')!=digest:entry['transformation']=str(entry['transformation'])+'; reviewed portability/evidence update'
    entry['export_sha256']=digest
for name,p in items.items():
    if name not in seen:record['files'].append({'source':None,'destination':name,'export_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'transformation':'Portable orchestration, frozen evidence or extraction validation'})
source.write_text(json.dumps(record,indent=2)+'\n')
items[source.relative_to(ROOT).as_posix()]=source
(ROOT/'provenance/files.sha256.json').write_text(json.dumps({name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(items.items())},indent=2)+'\n')
print('Frozen',len(items),'files')
