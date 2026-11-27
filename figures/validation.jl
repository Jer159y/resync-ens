# Read frozen validation ledgers only; no forecasts or selection changes.
using CairoMakie, DelimitedFiles, TOML, SHA
function tab(path)
 A,h=readdlm(path,',',Any,'\n';header=true)
 [Dict(Symbol(h[1,j])=>A[i,j] for j in axes(A,2)) for i in axes(A,1)]
end
f(x)=x isa Number ? Float64(x) : parse(Float64,string(x))
root="figures/panels";mkpath(root)
cases=[("Duffing oscillator","data/compact/main/duffing"),
 ("Charney–DeVore","data/compact/main/cdv"),
 ("Kuramoto–Sivashinsky","data/compact/main/ks200"),
 ("QG model (20 variables)","data/compact/main/rp20"),
 ("Cyclic Thomas (16 variables)","data/compact/main/thomas16")]
fig=Figure(size=(1300,850),fontsize=20);sources=Dict{String,String}()
for (i,(name,path)) in enumerate(cases)
 rows=tab(joinpath(path,"history.csv"));sel=TOML.parsefile(joinpath(path,"selection.toml"));scale=TOML.parsefile(joinpath(path,"scales.toml"))
 ax=Axis(fig[(i-1)÷3+1,(i-1)%3+1],title=name,xlabel="Validation W1 / fixed scale",ylabel="Validation spread loss / fixed scale",limits=(0,nothing,0,nothing))
 valid=filter(r->string(r[:valid])=="true",rows)
 xs=[f(r[:w1])/scale["w1"] for r in valid];ys=[f(r[:spread_loss])/scale["spread_loss"] for r in valid]
 scatter!(ax,xs,ys;color=:gray65,markersize=13)
 b=findfirst(r->f(r[:k])==0,valid);scatter!(ax,[xs[b]],[ys[b]];color="#B36145",marker=:rect,markersize=18)
 q=findfirst(r->f(r[:k])==sel["k"] && f(r[:multiplier])==sel["multiplier"],valid)
 scatter!(ax,[xs[q]],[ys[q]];color="#237E91",marker=:star5,markersize=26)
 text!(ax,"Selected: K=$(sel["k"]), m=$(sel["multiplier"])",position=(.04,.93),space=:relative,fontsize=17)
 for file in ["history.csv","selection.toml","scales.toml"]
  p=joinpath(path,file);sources[p]=bytes2hex(open(sha256,p))
 end
end
Legend(fig[3,1:3],[MarkerElement(color=:gray65,marker=:circle),MarkerElement(color="#B36145",marker=:rect),MarkerElement(color="#237E91",marker=:star5)], ["Evaluated valid setting","No correction","Selected scheme"],orientation=:horizontal,framevisible=false)
Label(fig[4,1:3],"Twelve corrected settings + baseline; two temporal validation origins. Linear axes.\nObserved evaluations only; this is not an exhaustive-grid or population Pareto front.",fontsize=17)
save(joinpath(root,"validation_selection.pdf"),fig);save(joinpath(root,"validation_selection.png"),fig;px_per_unit=1.2)
open(io->TOML.print(io,Dict("source_sha256"=>sources,"axes"=>"linear","scope"=>"Validation ledgers only")),joinpath(root,"validation_figure_provenance.toml"),"w")
println("VALIDATION_FIGURE_RENDERED")
