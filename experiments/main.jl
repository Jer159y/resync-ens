# julia --project=. -t 4 experiments/main.jl <case> smoke|test|bo|train
include("common.jl")
length(ARGS)>=2||error("Usage: main.jl duffing|cdv|ks200|rp20|thomas16 smoke|test|bo|train")
a=case_inputs(ARGS[1]);stage=ARGS[2]
function evaluate_candidate(a,c;role,origin=1,bases=200,tag="")
    ctx=context(a;role,origin,bases,m=c.multiplier)
    out=joinpath(RUN_ROOT,a.id,role,tag,"k$(c.k)_m$(c.multiplier)");mkpath(out)
    result=v4_policy_rollout(ctx,a.target,a.reference["spread_reference"],a.mu,a.sd,a.spec["base_tau"],out;
        k=c.k,m=c.multiplier,numerics=a.spec["numerics"],role=role)
    ctx=nothing;GC.gc();result,out
end
if stage=="smoke"
    ctx=context(a;bases=10)
    C=Vector{Matrix{Float64}}(undef,ctx.n);S=Vector{Any}(undef,ctx.n);G=fill(NaN,ctx.n,2)
    generate_candidates!(C,S,G,ctx,ctx.init_states,[copy(ctx.x0) for _ in 1:ctx.n],trues(ctx.n),false)
    hash=bytes2hex(sha256(reinterpret(UInt8,vec(cat(C...;dims=3)))))
    expected=TOML.parsefile(joinpath(a.root,"anchors/test/k0_m1/complete.toml"))["first_own_sha256"]
    hash==expected||error("First-own hash differs from preserved test: $hash != $expected")
    println("SMOKE_OK $(a.id): N=$(ctx.n), same first-own forecast, no state correction or selection")
elseif stage in ["test","prefix"]
    selected=TOML.parsefile(joinpath(a.root,"selection.toml"))
    for c in unique([(k=0,multiplier=1),(k=selected["k"],multiplier=selected["multiplier"])])
        result,out=evaluate_candidate(a,c;role="test",bases=stage=="prefix" ? 10 : 200,tag=stage=="prefix" ? "prefix" : "");result.valid||error("Invalid forecast recorded")
        check_anchor(a,out,c.k,c.multiplier;prefix=stage=="prefix")
    end
    println("$(uppercase(stage))_REPRODUCED $(a.id)")
elseif stage=="bo"
    # Re-execute the actual BO12 schedule, without accessing test targets.
    b=a.spec["bo"];domain=[(k=k,multiplier=m) for k in a.spec["design"]["correction_counts"] for m in a.spec["design"]["interval_multipliers"]]
    initial=[(k=x[1],multiplier=x[2]) for x in b["initial_design"]]
    function trial(c)
        ss=NamedTuple[]
        for (i,origin) in enumerate(a.spec["validation_origins"])
            r=try
                first(evaluate_candidate(a,c;role="validation",origin,bases=40,tag="origin$i"))
            catch err
                err isa V4NumericalForecastError||rethrow()
                (;valid=false,w1=Inf,spread_loss=Inf)
            end
            push!(ss,r)
        end
        (;c...,valid=all(s.valid for s in ss),w1=mean(s.w1 for s in ss),spread_loss=mean(s.spread_loss for s in ss))
    end
    base=trial((k=0,multiplier=1));ResyncV4BO.numeric(base)||error("Baseline invalid; do not force a corrected-only selection")
    tiny=a.spec["numerics"]["positive_loss_numerical_floor"]
    scales=(max(base.w1,a.reference["floor_w"],tiny),max(base.spread_loss,a.reference["floor_s"],tiny))
    history=NamedTuple[merge(base,(trial=0,method="paired_baseline",weight=NaN,acquisition=NaN,feasibility=NaN))]
    for i in 1:b["corrected_evaluations"]
        nxt=ResyncV4BO.next_candidate(history,domain,initial;scales,seed=b["bo_seed"],feasibility_ell=b["feasibility_length_scale"],explore_every=b["exploration_every_acquisition_steps"])
        push!(history,merge(trial(nxt.candidate),(trial=i,method=nxt.method,weight=nxt.weight,acquisition=nxt.acquisition,feasibility=nxt.feasibility)))
        paper1_csv(joinpath(RUN_ROOT,a.id,"validation/history.csv"),history)
    end
    # Frozen minimax and deterministic cost tie-break; no correction remains eligible.
    numeric=filter(ResyncV4BO.numeric,history)
    best=ResyncV4Metrics.select_scheme(history;floors=(max(a.reference["floor_w"],tiny),max(a.reference["floor_s"],tiny)),atol=a.spec["numerics"]["selection_absolute_tie_tolerance"],rtol=a.spec["numerics"]["selection_relative_tie_tolerance"]).selected
    scheme_toml(joinpath(RUN_ROOT,a.id,"validation/selection.toml"),Dict("k"=>best.k,"multiplier"=>best.multiplier,"test_accessed"=>false,"rule"=>"baseline_normalized_minimax"))
    println("BO_COMPLETE $(a.id): K=$(best.k), m=$(best.multiplier)")
elseif stage=="train"
    # Keep the preserved bank immutable. Retraining writes to a different directory.
    X=JLD2.load(joinpath(a.root,"training.jld2"),"traj");data=prepare_data(a.cfg,hcat(X,X[:,end]),1)
    base=esn_base_config(a.cfg);dir=joinpath(RUN_ROOT,a.id,"retrained_models");mkpath(dir)
    JLD2.jldsave(_ckpt_meta_path(dir);fingerprint=_ckpt_fingerprint(base,1234))
    Threads.@threads for i in 1:450
        f=_ckpt_member_path(dir,i)
        isfile(f)&&continue
        member=highdim_train(data.train_input,data.train_target,base,1234+i-1,a.spec["balanced_input"])
        _jldsave_quiet(f;ps=member[1],st=member[2])
    end
    println("RETRAIN_COMPLETE $(a.id): original artifacts unchanged")
else
    error("Unknown stage: $stage")
end
