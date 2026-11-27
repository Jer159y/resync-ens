"""Download the public input release; verify before extraction."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile,tempfile,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,help='Use an already downloaded archive');a=p.parse_args()
spec=json.loads((ROOT/'provenance/artifact.json').read_text());dest=ROOT/'artifacts'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
with tempfile.TemporaryDirectory(prefix='resync-inputs-') as tmp:
    if a.archive:archive=a.archive.resolve()
    else:
        url='https://github.com/Jer159y/resync-ens/releases/download/reproduction-inputs-v1/'+spec['name']
        request=urllib.request.Request(url,headers={'User-Agent':'resync-ens-reproduction'})
        with urllib.request.urlopen(request) as response, (Path(tmp)/spec['name']).open('wb') as target:
            shutil.copyfileobj(response,target,length=8*1024*1024)
        archive=Path(tmp)/spec['name']
    if digest(archive)!=spec['sha256']:raise SystemExit('Archive checksum mismatch; no files extracted')
    unpack=Path(tmp)/'unpacked';unpack.mkdir()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0]!='inputs' or not(member.isdir() or member.isfile()):raise SystemExit('Unsafe archive entry: '+member.name)
        tar.extractall(unpack)
    manifest=json.loads((ROOT/'provenance/inputs_manifest.json').read_text())
    for name,record in manifest.items():
        f=unpack/'inputs'/name
        if not f.is_file() or f.stat().st_size!=record['bytes'] or digest(f)!=record['sha256']:raise SystemExit('Input mismatch: '+name)
    if (dest/'inputs').exists():raise SystemExit('artifacts/inputs already exists; use a fresh clone to preserve your current inputs')
    dest.mkdir(exist_ok=True);shutil.move(str(unpack/'inputs'),str(dest/'inputs'))
print('Verified and installed',len(manifest),'input files in',dest/'inputs')
