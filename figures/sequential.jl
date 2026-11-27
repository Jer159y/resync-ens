# Re-layout the same six archived illustration traces. No fitting or forecasting.
using CairoMakie,DelimitedFiles,TOML,SHA
const OUT="figures/panels"
mkpath(OUT)
const SOURCES=Dict{String,String}()
function tab(path)
 SOURCES[path]=bytes2hex(open(sha256,path))
 a,h=readdlm(path,',',Any,'\n';header=true)
 Dict(Symbol(h[1,j])=>a[:,j] for j in axes(h,2))
end
num(x)=x isa Number ? Float64(x) : parse(Float64,string(x))
root="data/compact/sequential_reference"
np=joinpath(root,"normalization.toml");SOURCES[np]=bytes2hex(open(sha256,np));norm=TOML.parsefile(np)
ref=tab(joinpath(root,"reference_phase.csv"))
rq=num.(ref[:q]).*norm["population_sd"][1].+norm["mean"][1]
rv=num.(ref[:v]).*norm["population_sd"][2].+norm["mean"][2]
for (r,kind) in enumerate(["lstm","esn"]),(c,panel) in enumerate([1,3,6])
 t=tab("data/compact/sequence_model_examples_20261124/$(kind)_panel$(panel).csv")
 q=num.(t[:q]);v=num.(t[:v]);@assert all(x->-2<=x<=2,q)&&all(x->-1.5<=x<=1.5,v)
 f=Figure(size=(400,310),fontsize=26,figure_padding=0)
 ax=Axis(f;bbox=BBox(100,385,90,285),xlabel="Position q",ylabel=c==1 ? "Velocity v" : "",
  limits=(-2,2,-1.5,1.5),xticks=[-2,0,2],yticks=[-1,0,1],xticklabelsize=26,yticklabelsize=26,xlabelsize=26,ylabelsize=26,ylabelpadding=4)
 scatter!(ax,rq,rv;color=(:gray55,.2),markersize=2.5)
 lines!(ax,q,v;color=["#53616B","#B36145","#237A68"][c],linewidth=2.4)
 save(joinpath(OUT,"figure4_r$(r)_c$(c).pdf"),f)
end
open(joinpath(OUT,"provenance.toml"),"w") do io
 TOML.print(io,Dict("renderer_sha256"=>bytes2hex(open(sha256,@__FILE__)),"source_sha256"=>SOURCES,
 "new_forecasts"=>false,"panels"=>6,"q_limits"=>[-2.,2.],"v_limits"=>[-1.5,1.5],
 "panel_pixels"=>[400,310],"labels_and_layout"=>"LaTeX; model rows, condition columns"))
end
println("FIGURE4_LANDSCAPE_COMPLETE")
