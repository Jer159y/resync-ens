# Read-only panel exports. All panel letters, system/model/condition headings,
# legends and the multi-panel arrangements are supplied by LaTeX.
using CairoMakie,DelimitedFiles,TOML,SHA,Statistics
const OUT="figures/panels";mkpath(OUT)
const SOURCES=Dict{String,String}()
record(p)=(SOURCES[p]=bytes2hex(open(sha256,p));p)
num(x)=x isa Number ? Float64(x) : parse(Float64,string(x))
function tab(p)
 A,h=readdlm(record(p),',',Any,'\n';header=true)
 Dict(Symbol(h[1,j])=>A[:,j] for j in axes(h,2))
end
savepanel(name,f)=(save(joinpath(OUT,name*".pdf"),f);nothing)
using JLD2
ds=JLD2.load("data/compact/plot_inputs.jld2","datasets")
for (c,d) in enumerate(ds)
 X=hcat(d.R,d.B,d.C,reshape(d.truth,2,1));lo=vec(minimum(X;dims=2));hi=vec(maximum(X;dims=2));pad=0.07 .* max.(hi.-lo,eps());lim=(lo[1]-pad[1],hi[1]+pad[1],lo[2]-pad[2],hi[2]+pad[2])
 for row in 1:3
  f=Figure(size=(340,275),fontsize=24,figure_padding=0)
  ax=Axis(f;bbox=BBox(85,325,65,245),xlabel=row==1 ? "" : row==2 ? d.xlabel : d.cdf_label,ylabel=row<3 ? d.ylabel : c==1 ? "Probability" : "",xlabelsize=24,ylabelsize=24,xticklabelsize=23,yticklabelsize=23,ylabelpadding=4,xticks=WilkinsonTicks(3),yticks=row==3 ? [0,.5,1] : WilkinsonTicks(3))
  if row<3
   xlims!(ax,lim[1:2]...);ylims!(ax,lim[3:4]...)
   ax.xscale=d.signed ? CairoMakie.Makie.pseudolog10 : identity;ax.yscale=d.signed ? CairoMakie.Makie.pseudolog10 : identity
   row==1 && (ax.xticklabelsvisible=false;ax.xticksvisible=false)
   scatter!(ax,d.R[1,:],d.R[2,:];color=(:gray55,row==1 ? .6 : .12),markersize=row==1 ? 3.4 : 2.5)
   if row==2
    for (A,color) in [(d.B,"#B36145"),(d.C,"#237E91")]
     counts=Dict{Tuple{Float64,Float64},Int}();for xy in zip(A[1,:],A[2,:]);counts[xy]=get(counts,xy,0)+1;end
     xy=sort(collect(keys(counts)));scatter!(ax,first.(xy),last.(xy);color=(color,.65),markersize=[5.2sqrt(counts[p]) for p in xy])
    end
    scatter!(ax,[d.truth[1]],[d.truth[2]];marker=:diamond,color=:gray15,markersize=13,strokecolor=:white,strokewidth=.7)
   end
  else
   ylims!(ax,0,1)
   for ((x,p),color,style) in zip(d.cdf,[:gray55,"#B36145","#237E91"],[:dot,:solid,:dash]);lines!(ax,x,p;color,linestyle=style,linewidth=2.5);end
  end
  savepanel("figure2_r$(row)_c$(c)",f)
 end
end
cases=[(id,"data/compact/main/"*id,"data/compact/main/"*id) for id in ["duffing","cdv","ks200","rp20","thomas16"]]
for (r,(case,root,data)) in enumerate(cases)
 sel=TOML.parsefile(record(joinpath(root,"selection.toml")));k=sel["k"];m=sel["multiplier"]
 for c in 1:3
  f=Figure(size=(480,200),fontsize=22,figure_padding=0)
  ax=Axis(f;bbox=BBox(85,465,65,180),xlabel=r==5 ? "Base interval" : "",xlabelsize=23,xticklabelsize=21,yticklabelsize=21,limits=(0,200,0,nothing),xticks=0:50:200,yticks=WilkinsonTicks(3))
  for (kk,mm,color,style) in [(0,1,"#B36145",:solid),(k,m,"#237E91",:dash)]
   p=joinpath(root,"test","k$(kk)_m$(mm)",c==1 ? "effective_distribution_windows.csv" : "effective_base_windows.csv");t=tab(p)
   lines!(ax,num.(t[c==1 ? :stop_base : :base_index]),num.(t[c==1 ? :w1 : c==2 ? :spread : :crps]);color,linestyle=style,linewidth=2)
  end
  if c==2;ref=TOML.parsefile(record(joinpath(data,"reference.toml")))["spread_reference"];hlines!(ax,[ref];color="#53616B",linestyle=:dot,linewidth=1.4);end
  xlims!(ax,0,200);ax.limits=(0,200,0,nothing)
  savepanel("figure3_r$(r)_c$(c)",f)
 end
end

println("MAIN_PANELS_COMPLETE")
