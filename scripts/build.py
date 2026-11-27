"""Build the manuscript, optionally regenerating figures in a separate workspace."""
from pathlib import Path
import argparse,shutil,subprocess,os
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--redraw',action='store_true');p.add_argument('--assemble',action='store_true',help='Assemble existing panel PDFs without rerendering');a=p.parse_args();os.chdir(ROOT)
def run(cmd,cwd=ROOT):
    print('+',' '.join(map(str,cmd)),flush=True);subprocess.run(cmd,cwd=cwd,check=True)
figure_dir=ROOT/'paper/figures'
if a.redraw:
    render=ROOT/'build/render_workspace';render.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'figures',render/'figures',dirs_exist_ok=True)
    shutil.copytree(ROOT/'data/compact',render/'data/compact',dirs_exist_ok=True)
    for source in ['concept/draw.jl','main_panels.jl','exceedance.jl','sequential.jl','followups.jl','validation.jl','controls.jl']:
        run(['julia','--project='+str(ROOT),'--startup-file=no','figures/'+source],render)
if a.redraw or a.assemble:
    drawing_root=render if a.redraw else ROOT
    dest=ROOT/'build/figures';dest.mkdir(parents=True,exist_ok=True)
    for stem in ['fig01','fig02','fig03','fig04','figA01','figA02','figA03','figA04']:
        run(['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(dest),'figures/'+stem+'.tex'],drawing_root)
    figure_dir=dest
# Stage all outputs under build/; preserve immutable paper sources and archived figures.
work=ROOT/'build/paper';work.mkdir(parents=True,exist_ok=True)
for f in (ROOT/'paper').iterdir():
    if f.is_file() and f.suffix in ['.tex','.bib','.bbl','.cls','.cfg','.bst']:shutil.copy2(f,work/f.name)
(work/'figures').mkdir(exist_ok=True)
for stem in ['fig01','fig02','fig03','fig04','figA01','figA02','figA03','figA04']:
    shutil.copy2(figure_dir/(stem+'.pdf'),work/'figures'/(stem+'.pdf'))
run(['pdflatex','-interaction=nonstopmode','-halt-on-error','manuscript.tex'],work)
run(['bibtex','manuscript'],work)
for _ in range(2):run(['pdflatex','-interaction=nonstopmode','-halt-on-error','manuscript.tex'],work)
print('Built:',work/'manuscript.pdf')
