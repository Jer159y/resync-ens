using CairoMakie,DelimitedFiles,Statistics,SHA,TOML
root="data/compact/controls";figdir="figures/panels";mkpath(figdir)
A,h=readdlm(joinpath(root,"paired_event_contrasts.csv"),',',Any,'\n';header=true)
col=Dict(string(h[1,j])=>j for j in axes(h,2));num(x)=x isa Number ? Float64(x) : parse(Float64,string(x))
f=Figure(size=(1120,700),fontsize=18);axs=Axis[]
for (j,case) in enumerate(["duffing","ks200"]),(i,metric) in enumerate(["segment_marginal_w1","crps"])
 ax=Axis(f[i,j],title=case=="duffing" ? "Duffing" : "Kuramoto–Sivashinsky",xlabel="Parent-history event",ylabel=metric=="crps" ? "Paired CRPS difference" : "Paired segment W1 difference")
 push!(axs,ax)
 hlines!(ax,[0.];color=:gray50,linestyle=:dot)
 for (contrast,label,color) in [("full_minus_none","Resync − no intervention","#237A68"),("replay_given_boundary","Resync − boundary only","#B36145")]
  ix=[k for k in axes(A,1) if A[k,col["case"]]==case && A[k,col["metric"]]==metric && A[k,col["contrast"]]==contrast]
  lines!(ax,num.(A[ix,col["event"]]),num.(A[ix,col["paired_difference"]]);color=(color,.65),linewidth=1.2,label=label)
 end
end
Legend(f[3,:],first(axs);orientation=:horizontal,framevisible=false)
Label(f[4,:],"199 dependent events per system; descriptive paired contrasts, not independent-run confidence intervals.",fontsize=14)
save(joinpath(figdir,"paired_event_contrasts.pdf"),f);save(joinpath(figdir,"paired_event_contrasts.png"),f;px_per_unit=1.5)
open(joinpath(figdir,"provenance.toml"),"w") do io
 TOML.print(io,Dict("renderer_sha256"=>bytes2hex(open(sha256,@__FILE__)),"data_sha256"=>bytes2hex(open(sha256,joinpath(root,"paired_event_contrasts.csv"))),"linear_axes"=>true,"independent_run_CI"=>false))
end
