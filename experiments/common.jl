# Portable orchestration around the archived numerical kernels.
include(joinpath(@__DIR__,"../kernels/bootstrap.jl"))
const INPUT_ROOT=abspath(get(ENV,"RESYNC_ENS_INPUTS","artifacts/inputs"))
const RUN_ROOT=abspath(get(ENV,"RESYNC_ENS_RUNS","runs"))
function case_inputs(id)
    dir=joinpath(INPUT_ROOT,"cases",id)
    isfile(joinpath(dir,"case.toml"))||error("Missing input artifact for $id. Run python3 scripts/fetch_artifacts.py first.")
    spec=TOML.parsefile(joinpath(dir,"case.toml"));cfg=load_config(joinpath(dir,"config.toml"))
    cfg["system"]["data_file"]=joinpath(dir,"test.jld2")
    if haskey(cfg["system"],"reference_dir");cfg["system"]["reference_dir"]=joinpath(INPUT_ROOT,"qg");end
    if haskey(spec,"physics");spec["physics"]["reference_dir"]=joinpath(INPUT_ROOT,"qg");end
    q=JLD2.load(joinpath(dir,"target.jld2"))
    (id=id,root=dir,cfg=cfg,spec=spec,manifest=TOML.parsefile(joinpath(dir,"recorded_data_manifest.toml")),
     reference=TOML.parsefile(joinpath(dir,"reference.toml")),target=q["target"],mu=q["mu"],sd=q["sd"],bank=joinpath(dir,"models"))
end
function context(a;role="test",origin=1,bases=200,m=1,bank=a.bank,seed=1234,cohort_file=nothing)
    base=a.spec["base_tau"];offset=role=="test" ? a.spec["sync_len"]+1 : origin
    datafile=joinpath(a.root,role=="test" ? "test.jld2" : "validation.jld2")
    out=joinpath(RUN_ROOT,a.id,role,"context_m$m");mkpath(out)
    all(isfile(_ckpt_member_path(bank,i)) for i in 1:450)||error("Incomplete preserved model bank")
    cfg=deepcopy(a.cfg);cfg["system"]["data_file"]=datafile
    ctx=build_resync_context(sys_from_config(cfg,datafile);n_ensemble=450,τ=m*base,n_segments=bases÷m,
        checkpoint_dir=bank,log_dir=out,log_name="context",seed_start=seed,test_offset=offset,
        sync_len=min(a.spec["sync_len"],offset-1),need_geometry=false)
    if cohort_file===nothing
        cohort_file=joinpath(a.root,"cohorts",role=="test" ? "test.toml" : "validation_origin$(findfirst(==(origin),a.spec["validation_origins"])).toml")
    end
    if isfile(cohort_file)
        ids=TOML.parsefile(cohort_file)["member_ids"]
    else
        ids,info=refill_until_clean(ctx,300,:feature_distance_from_centroid;p=.1,random_seed=777)
        info.converged||error("Initial refill failed: preserve failure, do not substitute seeds")
    end
    length(ids)==length(unique(ids))==300||error("Cohort changed")
    subset_context(ctx,ids)
end
function check_anchor(a,out,k,m;atol=1e-9,prefix=false)
    ref=joinpath(a.root,"anchors/test/k$(k)_m$(m)")
    isdir(ref)||error("No original anchor for this policy")
    expected=TOML.parsefile(joinpath(ref,"complete.toml"))["first_own_sha256"]
    actual=TOML.parsefile(joinpath(out,"complete.toml"))["first_own_sha256"]
    actual==expected||error("Original initialization was not reproduced")
    worst=0.
    for scope in ["own","effective"],(file,keys) in [("base_windows.csv",["spread","spread_loss","crps"]),("distribution_windows.csv",["w1"])]
        function table(p)
            values,headers=readdlm(p,',',Any,'\n';header=true);Dict(string(headers[1,j])=>Float64.(values[:,j]) for j in eachindex(headers) if string(headers[1,j]) in keys)
        end
        x=table(joinpath(ref,scope*"_"*file));y=table(joinpath(out,scope*"_"*file))
        for key in keys
            n=length(y[key]);(prefix ? n<=length(x[key]) : n==length(x[key]))||error("Window count mismatch")
            worst=max(worst,maximum(abs.(x[key][1:n].-y[key])))
        end
    end
    worst<=atol||error("Anchor discrepancy $worst exceeds $atol")
    scheme_toml(joinpath(out,"reproduction_check.toml"),Dict("first_own_match"=>true,"maximum_metric_difference"=>worst,"tolerance"=>atol,"prefix_only"=>prefix))
end
