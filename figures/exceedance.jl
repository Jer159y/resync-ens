# Existing snapshot and calibration points only. No model/forecast execution.
using CairoMakie,DelimitedFiles,TOML,SHA
const ROOT="figures/panels";mkpath(ROOT)
const INPUT="data/compact/figure2_reference_exceedance_20261125/exceedance.csv"
num(x)=x isa Number ? Float64(x) : parse(Float64,string(x))
A,h=readdlm(INPUT,',',Any,'\n';header=true)
rows=[Dict(Symbol(h[1,j])=>A[i,j] for j in axes(A,2)) for i in axes(A,1)]
for (c,case) in enumerate(["duffing","cdv","ks200","rp20","thomas16"])
 f=Figure(size=(340,275),fontsize=24,figure_padding=0)
 ax=Axis(f;bbox=BBox(85,325,65,245),xlabel="Reference interval (%)",ylabel=c==1 ? "Outside (%)" : "",xlabelsize=23,ylabelsize=24,xticklabelsize=23,yticklabelsize=23,ylabelpadding=4,xticks=[90,95,99],yticks=[0,20,40,60],xgridcolor=(:gray60,.15),ygridcolor=(:gray60,.15))
 ymax=0.
 for (policy,color,style,marker) in [("reference_check","#53616B",:dot,:rect),("no_correction","#B36145",:solid,:circle),("selected","#237E91",:dash,:utriangle)]
  q=sort(filter(r->r[:case]==case&&r[:policy]==policy,rows);by=r->num(r[:marginal_interval_percent]))
  x=num.([r[:marginal_interval_percent] for r in q]);y=num.([r[:outside_percent] for r in q]);ymax=max(ymax,maximum(y))
  lines!(ax,x,y;color,linestyle=style,linewidth=3)
  scatter!(ax,x,y;color,marker,markersize=8)
 end
 @assert ymax<=70
 xlims!(ax,89,100);ylims!(ax,0,70)
 save(joinpath(ROOT,"reference_exceedance_c$(c).pdf"),f)
end
open(joinpath(ROOT,"provenance.toml"),"w") do io
 TOML.print(io,Dict("renderer_sha256"=>bytes2hex(open(sha256,@__FILE__)),"input_sha256"=>bytes2hex(open(sha256,INPUT)),"panels"=>5,"new_forecasts"=>false,"axis_scale"=>"linear; shared percentage y axes 0–70%; same snapshot as rows1/2","percentage_y_limits"=>[0,70],"percentage_y_ticks"=>[0,20,40,60]))
end
println("REFERENCE_EXCEEDANCE_PANELS_COMPLETE")
