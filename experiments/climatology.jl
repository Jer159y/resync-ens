include("common.jl")
include(joinpath(@__DIR__,"../kernels/climatology.jl"))
function main()
a=case_inputs(first(ARGS));base=a.spec["base_tau"];off=a.spec["training_prefix_length"];origin=a.spec["sync_len"]+1
raw=JLD2.load(joinpath(a.root,"test.jld2"),"traj");Y=(raw[:,off+origin+1:off+origin+200base].-a.mu)./a.sd
D=size(Y,1);bybase=zeros(200);dispersion=0.;stride=max(1,base÷20)
for d in 1:D
    f=prepare_distribution(view(a.target,d,:));dispersion+=f.dispersion/D
    for b in 1:200
        bybase[b]+=mean(score_distribution(f,Y[d,(b-1)*base+t]) for t in 1:stride:base)/D
    end
end
out=joinpath(RUN_ROOT,a.id,"climatology");mkpath(out)
rows=[(system=a.id,base_index=b,calibration_cdf_crps=bybase[b],expected_iid300_crps=bybase[b]+dispersion/300) for b in 1:200]
paper1_csv(joinpath(out,"base_windows.csv"),rows)
A,h=readdlm(joinpath(@__DIR__,"../data/compact/manuscript_review_20261123/$(a.id)_climatology_base.csv"),',',Any;header=true)
maximum(abs.(Float64.(A[:,3]).-bybase))<1e-12||error("Climatology anchor mismatch")
println("CLIMATOLOGY_REPRODUCED $(a.id)")

end
main()
