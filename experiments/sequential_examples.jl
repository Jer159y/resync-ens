# Reconstruct the two published selected examples directly from archived model weights.
using TOML,DelimitedFiles
root=normpath(joinpath(@__DIR__,".."));kind=first(ARGS)
kind in ["lstm","esn"]||error("Usage: sequential_examples.jl lstm|esn")
job=TOML.parsefile(joinpath(root,"configs/sequential_$kind.toml"));job["output"]="runs/unused";job["model_root"]="unused"
out=abspath(get(ENV,"RESYNC_ENS_RUNS","runs"));mkpath(out);path=joinpath(out,"example_profile.toml")
open(io->TOML.print(io,job),path,"w");ARGS[1]=path
include(joinpath(root,"kernels/sequential_validation.jl"))
inputs=abspath(get(ENV,"RESYNC_ENS_INPUTS","artifacts/inputs"))
member,donor=kind=="lstm" ? (145,162) : (32,41)
m=deserialize(joinpath(inputs,"sequential",kind,"models","member_$member.bin"))
d=deserialize(joinpath(inputs,"sequential",kind,"models","member_$donor.bin"))
raw,_=trajectory(8000,2026112451);mu=mean(raw[:,1:2000];dims=2);sd=std(raw[:,1:2000];dims=2,corrected=false)
V,_=trajectory(19000,2026112462);Y=(V.-mu)./sd;origin=1001;t=2000;x0=Y[:,origin]
saved=drive2(m,zero_state(m),Y[:,origin-1000:origin-1]);dsaved=drive2(d,zero_state(d),Y[:,origin-1000:origin-1])
P,_=rollout2(m,saved,x0,4000);D,_=rollout2(d,dsaved,x0,4000)
old=drive2(m,saved,hcat(x0,P[:,1:t-1]));rep=drive2(m,saved,hcat(x0,D[:,1:t-1]))
none,_=rollout2(m,old,P[:,t],1000);resync,_=rollout2(m,rep,D[:,t],1000)
for (panel,A) in [(1,P[:,1:1000]),(3,none),(6,resync)]
    Q=A.*sd.+mu;reference=readdlm(joinpath(root,"data/compact/sequence_model_examples_20261124/$(kind)_panel$panel.csv"),',',Float64;skipstart=1)
    maximum(abs.(permutedims(Q[:,1:2:1000]).-reference[:,2:3]))<1e-10||error("Published trace differs: $kind panel $panel")
end
println("EXAMPLE_TRACES_REPRODUCED $kind: all three conditions agree with published coordinates")
