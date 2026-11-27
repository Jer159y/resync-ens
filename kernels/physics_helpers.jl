# Extracted without numerical edits from tools/run_scheme_bo.jl; see provenance/source_export.json.
scheme_sha(path)=bytes2hex(open(sha256,path))
function scheme_toml(path, value)
    mkpath(dirname(path));tmp=path*".tmp"
    open(io->TOML.print(io,value;sorted=true),tmp,"w");mv(tmp,path;force=true)
end

"Same physical equation/integrator step; fresh Duffing initial condition, phase zero."
function scheme_duffing(cfg,n,seed,transient)
    p=cfg["system"];rng=MersenneTwister(seed);x=0.1.*randn(rng,2)
    dt=p["dt"];h=p["int_dt"];sub=round(Int,dt/h);t=0.
    f(x,t)=[x[2],-p["delta"]*x[2]+p["alpha"]*x[1]-p["beta"]*x[1]^3+p["gamma"]*cos(p["omega"]*t)]
    X=Matrix{Float64}(undef,2,n)
    for j in 1:transient+n
        for _ in 1:sub
            a=f(x,t);b=f(x.+h/2 .*a,t+h/2);c=f(x.+h/2 .*b,t+h/2);e=f(x.+h.*c,t+h)
            x .+= h/6 .* (a.+2 .*b.+2 .*c.+e);t+=h
        end
        j>transient && (X[:,j-transient].=x)
    end
    all(isfinite,X)||error("Nonfinite fresh Duffing trajectory")
    X
end

function scheme_fresh(cfg,n,seed,d)
    if cfg["system"]["name"]=="duffing"
        return scheme_duffing(cfg,n,seed,d["duffing_transient"])
    end
    fresh=deepcopy(cfg);fresh["system"]["seed"]=seed;fresh["system"]["n_saved"]=n
    X,_,_,_=generate_system_data(fresh)
    X
end

function scheme_joint(root,phase,cfg,source,n,seed,d)
    path=joinpath(root,phase*"_data.jld2")
    isfile(path)&&return path
    println("GENERATE $phase seed=$seed saved=$n (fresh physical initial condition)");flush(stdout)
    fresh=scheme_fresh(cfg,n,seed,d);off=cfg["data"]["shift"]+cfg["data"]["train_len"]
    original=JLD2.load(source,"traj");prefix=original[:,1:off];original=nothing
    atomic_save(path;traj=hcat(prefix,fresh),physical_seed=seed,training_source_sha256=scheme_sha(source),
        training_prefix_length=off,fresh_start=off+1)
    path
end

