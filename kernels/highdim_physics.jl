# Extracted without numerical edits from tools/highdim_v4_physics_20261123.jl; see provenance/source_export.json.
function highdim_physical_files(physics)
    physics["name"]=="rp20" ? [joinpath(physics["reference_dir"],f) for f in ["metadata.toml","tensor.csv"]] : String[]
end
function highdim_fresh(cfg,n,seed,physics;half=false)
    D=cfg["system"]["N"];h=physics["internal_dt"]/(half ? 2 : 1)
    cfg["system"]["name"]=="file" || error("Preserved file-training configuration required")
    if physics["name"]=="rp20"
        m=LowOrderQG.load_model(physics["reference_dir"])
        D==20 && m.dimension==20 || error("Full20-variable QG state required")
        bytes2hex(open(sha256,joinpath(physics["reference_dir"],"tensor.csv")))==m.metadata["tensor_sha256"] || error("QG tensor hash mismatch")
        return LowOrderQG.trajectory(m;n,dt=physics["dt"],h,transient=physics["transient_time"],seed,lyap_steps=0).traj
    end
    physics["name"]=="thomas16" && D==16 && physics["parameter"]==.18 || error("Frozen cyclic16-variable equation required")
    sub=round(Int,physics["dt"]/h);isapprox(sub*h,physics["dt"]) || error("Integrator/save step mismatch")
    # Original initial-state distribution, now with a separately frozen seed per role.
    x=randn(MersenneTwister(seed),D)./sqrt(2physics["parameter"]);w=qhd_workspace(D)
    for _ in 1:round(Int,physics["transient_time"]/h);qhd_step!(x,h,"thomas16",physics["parameter"],w);end
    X=Matrix{Float64}(undef,D,n)
    for t in 1:n
        for _ in 1:sub;qhd_step!(x,h,"thomas16",physics["parameter"],w);end
        X[:,t].=x
    end
    all(isfinite,X) || error("Nonfinite physical trajectory")
    X
end

# Same type name and implementation as setting_hunt_common.jl, needed to load original ESNs.
struct HuntBalancedInput
    scale::Float64
end
function (f::HuntBalancedInput)(rng,dims::Integer...)
    length(dims)==2 || error("Expected reservoir-by-input matrix")
    N,D=dims;columns=shuffle(rng,[mod1(i,D) for i in 1:N])
    sparse(collect(1:N),columns,f.scale.*(2rand(rng,N).-1),N,D)
end
function hunt_train_one(input,target,base,seed)
    rng=MersenneTwister(seed)
    esn=ESN(base.in_size,base.res_size,base.out_size,tanh;
        init_reservoir=rand_sparse(;radius=base.radius,sparsity=base.sparsity,return_sparse=true),
        init_input=HuntBalancedInput(base.scaling),state_modifiers=base.state_modifiers,
        leak_coefficient=base.leak_coefficient,init_state=(rng,dims...)->randn(rng,Float32,dims...),
        use_bias=true,init_bias=TestUtils.bias_initializer(base),readout_activation=base.readout_activation)
    ps,st=ReservoirComputing.setup(rng,esn)
    ps,st=ReservoirComputing.train!(esn,input,target,ps,st,base.training_method;washout=base.washout,solver=base.solver)
    ps,st,esn
end

highdim_train(input,target,base,seed,balanced)=balanced ? hunt_train_one(input,target,base,seed) : _train_one_member(input,target,base,seed;sparse=true)
