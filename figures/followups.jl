# Plot existing compact outcomes only. Panel letters and headings belong to LaTeX.
using CairoMakie,DelimitedFiles,TOML,SHA
const OUT="figures/panels";mkpath(OUT)
const INPUT="data/compact/followup_results_20261125"
function rows(file)
 A,h=readdlm(joinpath(INPUT,file),',',Any,'\n';header=true)
 [Dict(Symbol(h[1,j])=>A[i,j] for j in axes(A,2)) for i in axes(A,1)]
end
num(x)=x isa Number ? Float64(x) : parse(Float64,string(x))
const BASE="#B36145";const RESYNC="#237E91";const REF="#53616B"
for case in ["duffing","ks200"]
history=filter(x->x[:case]==case,rows("history.csv"))
for (c,key) in enumerate([:w1,:crps,:forecast_seconds])
 f=Figure(size=(400,290),fontsize=26,figure_padding=0)
 ax=Axis(f;bbox=BBox(110,385,80,265),xlabel="Recent length L / τ₀",ylabel=c==1 ? "Returned W1" : c==2 ? "Returned CRPS" : "Rollout time (10³ s)",
  xlabelsize=24,ylabelsize=24,xticklabelsize=23,yticklabelsize=23,ylabelpadding=5,xticks=[0,64,128],yticks=c==3 ? (case=="duffing" ? [0,.5,1.] : [0,5,10]) : WilkinsonTicks(3),xgridcolor=(:gray60,.15),ygridcolor=(:gray60,.15))
 q=sort(filter(x->startswith(string(x[:policy]),"recent_steps_")||x[:policy]=="boundary",history);by=x->num(x[:history_in_base]))
 scale=c==3 ? 1000. : 1.
 xx=num.([x[:history_in_base] for x in q]);yy=num.([x[key] for x in q])./scale
 lines!(ax,xx,yy;color=BASE,linewidth=3.8);scatter!(ax,xx,yy;color=BASE,markersize=8)
 for (pol,color,style) in [("none",REF,:dot),("full",RESYNC,:dash)]
  value=num(only(filter(x->x[:policy]==pol,history))[key])/scale;hlines!(ax,[value];color,linestyle=style,linewidth=2.8)
 end
 ymax=maximum(vcat(yy,[num(x[key])/scale for x in history]))
 xlims!(ax,-2,130);ax.limits=(-2,130,-.035ymax,1.07ymax)
 if c<3
  # Explicit linear zoom of the short-history plateau, retaining the full-range main axes.
  inset=Axis(f;bbox=BBox(228,377,145,250),xticks=case=="duffing" ? [.1,.3,.5] : [0,.075,.15],yticks=case=="duffing" ? (c==1 ? [.025,.03] : [.564,.568]) : (c==1 ? [.08,.09] : [.576,.578]),
   xticklabelsize=18,yticklabelsize=18,xgridvisible=false,ygridvisible=false,backgroundcolor=:white)
  value=num(only(filter(x->x[:policy]=="full",history))[key])
  hlines!(inset,[value];color=RESYNC,linestyle=:dash,linewidth=2.5)
  lines!(inset,xx,yy;color=BASE,linewidth=2.8);scatter!(inset,xx,yy;color=BASE,markersize=5.5)
  xlims!(inset,case=="duffing" ? (.1,.5) : (0.,.15));ylims!(inset,case=="duffing" ? (c==1 ? (.024,.031) : (.5635,.569)) : (c==1 ? (.080,.092) : (.575,.580)))
 end
 save(joinpath(OUT,"figure5_$(case)_c$(c).pdf"),f)
end
end
for case in ["duffing","ks200"]
rep=filter(x->x[:case]==case,rows("replication.csv"))
for (c,key) in enumerate([:w1,:spread,:crps])
 f=Figure(size=(400,290),fontsize=26,figure_padding=0)
 ax=Axis(f;bbox=BBox(110,385,80,265),ylabel="Ensemble realization",xlabel=c==1 ? "Returned W1" : c==2 ? "Spread S" : "Returned CRPS",
  xlabelsize=24,ylabelsize=24,xticklabelsize=23,yticklabelsize=23,ylabelpadding=5,yticks=([1,2,3],["E1","E2","E3"]),xticks=WilkinsonTicks(3),yreversed=true,xgridcolor=(:gray60,.15),ygridcolor=(:gray60,.15))
 c==2 && vlines!(ax,[1.];color=REF,linestyle=:dash,linewidth=3)
 for (b,seed) in enumerate([310000,320000,330000]),(j,phys) in enumerate([2026112401,2026112402])
  q=[only(filter(x->num(x[:bank_seed_start])==seed&&num(x[:physical_seed])==phys&&x[:policy]==pol,rep)) for pol in ["none","full"]]
  center=b+(j==1 ? -.17 : .17);y=[center,center];x=num.([v[key] for v in q])
  lines!(ax,x,y;color=(:gray40,.9),linewidth=2.8)
  for i in 1:2
   expanded=string(q[i][:expanded_reserve])=="True"
   scatter!(ax,[x[i]],[y[i]];color=expanded ? :white : [BASE,RESYNC][i],strokecolor=[BASE,RESYNC][i],strokewidth=2.4,
    marker=j==1 ? :circle : :utriangle,markersize=13)
  end
 end
 ymax=maximum(num(x[key]) for x in rep)
 ax.limits=(-.055ymax,1.08ymax,.5,3.5)
 if case=="ks200" && c==3
  # Explicit linear zoom: paired CRPS differences are small relative to the score.
  ax.xticks=[.575,.585,.595];ax.limits=(.575,.595,.5,3.5)
 end
 save(joinpath(OUT,"figure6_$(case)_c$(c).pdf"),f)
end
end
sources=Dict(joinpath(INPUT,p)=>bytes2hex(open(sha256,joinpath(INPUT,p))) for p in ["history.csv","replication.csv","verification.json"])
open(joinpath(OUT,"provenance.toml"),"w") do io
 TOML.print(io,Dict("renderer_sha256"=>bytes2hex(open(sha256,@__FILE__)),"source_sha256"=>sources,
  "panel_counts"=>Dict("figure5_duffing"=>3,"figure6_duffing"=>3,"figure5_ks200"=>3,"figure6_ks200"=>3),"ks_plotted"=>true,"labels_and_layout"=>"LaTeX","new_forecasts"=>false))
end
println("FOLLOWUP_PANELS_COMPLETE")
