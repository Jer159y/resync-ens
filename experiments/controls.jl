# Same frozen cohort; 199 common-parent event forks, not 199 independent runs.
include("common.jl")
include(joinpath(@__DIR__,"../kernels/matched_controls.jl"))
id=first(ARGS);id in ["duffing","ks200"]||error("Archived control design covers Duffing and KS200")
a=case_inputs(id);bases="--smoke" in ARGS ? 10 : 200
k=id=="duffing" ? 30 : 60
for r in 0:1,b in 0:1
    ctx=context(a;bases)
    out=joinpath(RUN_ROOT,id,"controls","b$(b)_r$(r)")
    control_rollout(ctx,a,Dict("numerics"=>a.spec["numerics"]),out;k,m=1,boundary=b==1,replay=r==1,
        forks=b==1&&r==1,original=joinpath(a.root,"anchors"),case=id,bases)
    ctx=nothing;GC.gc()
end
