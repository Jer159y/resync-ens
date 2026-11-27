# Fixed postselection history-length sweep; never reselect K or m from these tests.
include("common.jl")
include(joinpath(@__DIR__,"../kernels/matched_controls.jl"))
const REVCASE=first(ARGS)
REVCASE in ["duffing","ks200"]||error("Only the archived Duffing/KS sweep is specified")
const REVP=TOML.parsefile(joinpath(@__DIR__,"../configs/resync_revision_20261124.toml"))
source=joinpath(@__DIR__,"../kernels/followups.jl");s=read(source,String)
for (old,new) in [("W=policy==\"recent_quarter\" ?","W=startswith(policy,\"recent_steps_\") ? parse(Int,split(policy,\"_\")[end]) : policy==\"recent_quarter\" ?"),("replay=policy in [","replay=startswith(policy,\"recent_steps_\") || policy in ["),("boundary=policy in [","boundary=startswith(policy,\"recent_steps_\") || policy in [")]
    length(findall(old,s))==1||error("Numerical source adapter no longer matches");s=replace(s,old=>new)
end
include_string(Main,s,source)
a=case_inputs(REVCASE);p=TOML.parsefile(joinpath(@__DIR__,"../configs/resync_history_dense_20261124.toml"))
base=a.spec["base_tau"];bases="--smoke" in ARGS ? 10 : 200
steps=sort(unique(vcat([1],max.(1,round.(Int,base.*p["history_lengths_in_base_intervals"])))))
"--smoke" in ARGS&&(steps=[1,max(1,round(Int,base/4))])
for policy in vcat(["none","boundary"],["recent_steps_$w" for w in steps],["full"])
    ctx=context(a;bases)
    revision_rollout(ctx,a,Dict("numerics"=>a.spec["numerics"]),joinpath(RUN_ROOT,REVCASE,"history",policy);policy,bases,base)
    ctx=nothing;GC.gc()
end
