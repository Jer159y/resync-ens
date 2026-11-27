# Reproduce the crossed 3 banks x 2 physical starts; no fresh parameter selection.
include("common.jl")
include(joinpath(@__DIR__,"../kernels/matched_controls.jl"))
const REVCASE=first(ARGS)
REVCASE in ["duffing","ks200"]||error("Archived crossed replication covers Duffing and KS200")
const REVP=TOML.parsefile(joinpath(@__DIR__,"../configs/resync_revision_20261124.toml"))
include(joinpath(@__DIR__,"../kernels/followups.jl"))
a=case_inputs(REVCASE);base=a.spec["base_tau"];bases=200
prefix=JLD2.load(joinpath(a.root,"training.jld2"),"traj");data=prepare_data(a.cfg,hcat(prefix,prefix[:,end]),1);cfg=esn_base_config(a.cfg)
for seed in REVP["new_bank_seed_starts"]
    # The third Duffing bank required a separately disclosed larger reserve cap.
    cap=REVCASE=="duffing" && seed==330000 ? 750 : 450
    bank=joinpath(RUN_ROOT,REVCASE,"replication","models","seed$seed");mkpath(bank)
    isfile(_ckpt_meta_path(bank))||JLD2.jldsave(_ckpt_meta_path(bank);fingerprint=_ckpt_fingerprint(cfg,seed))
    Threads.@threads for i in 1:cap
        file=_ckpt_member_path(bank,i);isfile(file)&&continue
        member=_train_one_member(data.train_input,data.train_target,cfg,seed+i-1;sparse=true)
        _jldsave_quiet(file;ps=member[1],st=member[2])
    end
    for physical in REVP["new_physical_seeds"]
        root=joinpath(RUN_ROOT,REVCASE,"replication","bank$(seed)_physical$(physical)");mkpath(root)
        file=joinpath(root,"physical_data.jld2");origin=a.spec["sync_len"]+1
        if !isfile(file)
            fresh=scheme_fresh(a.cfg,origin+bases*base,physical,a.spec["physical_design"])
            atomic_save(file;traj=hcat(prefix,fresh),physical_seed=physical,training_prefix_unchanged=true)
        end
        allctx=build_resync_context(sys_from_config(a.cfg,file);n_ensemble=cap,τ=base,n_segments=bases,checkpoint_dir=bank,
            log_dir=root,log_name="replication",seed_start=seed,test_offset=origin,sync_len=origin-1,need_geometry=false)
        ids,info=refill_until_clean(allctx,300,:feature_distance_from_centroid;p=.1,max_rounds=50,random_seed=777)
        info.converged||error("Refill failed; no automatic seed/budget replacement")
        scheme_toml(joinpath(root,"cohort.toml"),Dict("member_ids"=>ids,"seed_start"=>seed,"physical_seed"=>physical,"reserve_used"=>info.reserve_used,"total_model_cap"=>cap))
        for policy in ["none","full"]
            ctx=subset_context(allctx,ids)
            revision_rollout(ctx,a,Dict("numerics"=>a.spec["numerics"]),joinpath(root,policy);policy,bases,base)
        end
        allctx=nothing;GC.gc()
    end
end
