# Generic role-aware streaming rollout, derived from the verified immutable smoke executor.
struct V4NumericalForecastError <: Exception
    reason::String
end
function v4_policy_rollout(ctx,target,sref,mu,sd,base,out;k=0,m=1,numerics,role="validation")
    role in ("validation","test") || error("Unknown evaluation role")
    complete=joinpath(out,"complete.toml")
    if isfile(complete)
        TOML.parsefile(complete)["evaluation_role"]==role || error("Cached role mismatch")
        return JLD2.load(joinpath(out,"scores.jld2"),"summary")
    end
    mkpath(out);tau=ctx.τ;N=ctx.n;D=length(ctx.x0);T=ctx.predict_len
    tau==m*base && T%tau==0 || error("Intervention/score grid mismatch")
    kwargs=(probability_stride=max(1,base÷20),dimension_samples=numerics["effective_dimension_sample_times"],
        constant_tol=numerics["constant_amplitude_tolerance"])
    own_eval=WindowEvaluator(base,N,T÷base,target,sref;kwargs...)
    eff_eval=WindowEvaluator(base,N,T÷base,target,sref;kwargs...)
    mu3=reshape(mu,D,1,1);sd3=reshape(sd,D,1,1)
    to_eval(A)=(A.*reshape(ctx.norm.sd,D,1,1).+reshape(ctx.norm.mu,D,1,1).-mu3)./sd3
    truth_eval=(ctx.truth.*ctx.norm.sd.+ctx.norm.mu.-mu)./sd
    states=Any[deepcopy(st) for st in ctx.init_states];inputs=[copy(ctx.x0) for _ in 1:N]
    corrected=k>0
    hist=corrected ? [Matrix{Float64}(undef,D,T) for _ in 1:N] : Matrix{Float64}[]
    src=zeros(Int,N,ctx.n_segments)
    opts=ResyncOptions(p=corrected ? k/N : .1,bad_selector=:feature_distance_from_centroid,donor=:uniform)
    cands=Vector{Matrix{Float64}}(undef,N);next_states=Vector{Any}(undef,N);geom=fill(NaN,N,2)
    firsthash="";events=NamedTuple[];started=time();forecast_seconds=0.
    for s in 1:ctx.n_segments
        tick=time()
        generate_candidates!(cands,next_states,geom,ctx,states,inputs,trues(N),false)
        A=cat(cands...;dims=3)
        if s==1
            firsthash=bytes2hex(sha256(reinterpret(UInt8,vec(Array(view(A,:,1:base,:))))))
        end
        fin=[all(isfinite,c) for c in cands]
        bad=Int[];donors=Dict{Int,Int}()
        if corrected
            sc=criterion_scores(:feature_distance_from_centroid,SegmentPool(cands,fin,geom,s,opts.random_seed),ctx)
            bad=select_bad(sc,fin,trues(N),k)
            badset=Set(bad);good=[i for i in 1:N if !(i in badset)]
            isempty(good) && throw(V4NumericalForecastError("No finite nonselected donor"))
            donors=assign_donors(bad,good,opts,s)
        end
        states=Any[deepcopy(st) for st in next_states];inputs=[copy(c[:,end]) for c in cands]
        B=corrected ? copy(A) : A
        if corrected
            cols=seg_range(s,tau)
            for i in 1:N;hist[i][:,cols].=cands[i];src[i,s]=i;end
            before=copy(src[:,1:s])
            for j in bad
                d=donors[j]
                hist[j][:,cols].=cands[d]
                src[j,1:(s-1)].=view(before,d,1:(s-1));src[j,s]=j
                B[:,:,j].=cands[d]
            end
            Threads.@threads for j in bad
                Y=assemble_history(hist,src,j,s,tau);ps,_,esn=ctx.ensemble[j]
                states[j]=warmup_full_history(ps,ctx.init_states[j],esn,Y;x0=ctx.x0)
                inputs[j]=copy(Y[:,end])
            end
        end
        forecast_seconds+=time()-tick
        AZ=to_eval(A);BZ=corrected ? to_eval(B) : AZ
        for lo in 1:base:tau
            range=lo:(lo+base-1);global_range=((s-1)*tau+lo):((s-1)*tau+lo+base-1)
            y=view(truth_eval,:,global_range)
            push_base!(own_eval,view(AZ,:,range,:),y);push_base!(eff_eval,view(BZ,:,range,:),y)
        end
        push!(events,(segment=s,n_updated=length(bad),nonfinite_own_members=count(!,fin)))
        println("V4 $role k=$k m=$m segment=$s/$(ctx.n_segments)");flush(stdout)
    end
    summary=merge(finish(eff_eval),(k=k,multiplier=m,own_valid=finish(own_eval).valid,
        own_effective_dimension=finish(own_eval).effective_dimension,first_own_sha256=firsthash,
        forecast_seconds=forecast_seconds,end_to_end_seconds=time()-started))
    scalar_rows(rows)=map(rows) do r
        (;(key=>value for (key,value) in pairs(r) if !(value isa AbstractArray))...)
    end
    for (scope,e) in (("own",own_eval),("effective",eff_eval))
        paper1_csv(joinpath(out,"$(scope)_base_windows.csv"),scalar_rows(e.base_rows))
        paper1_csv(joinpath(out,"$(scope)_distribution_windows.csv"),e.distribution_rows)
        coord=[(scope=scope,base_index=r.base_index,coordinate=d,coverage50=r.coverage50_by_coordinate[d],
            coverage90=r.coverage90_by_coordinate[d],n_scored=r.n_scored_by_coordinate[d]) for r in e.base_rows for d in 1:D]
        paper1_csv(joinpath(out,"$(scope)_coverage_coordinates.csv"),coord)
    end
    paper1_csv(joinpath(out,"events.csv"),events);paper1_csv(joinpath(out,"summary.csv"),[summary])
    atomic_save(joinpath(out,"scores.jld2");summary=summary)
    scheme_toml(complete,Dict("status"=>"complete","k"=>k,"multiplier"=>m,
        "valid"=>summary.valid,"end_to_end_seconds"=>summary.end_to_end_seconds,
        "forecast_seconds"=>forecast_seconds,"first_own_sha256"=>firsthash,"test_accessed"=>(role=="test"),"evaluation_role"=>role))
    summary
end
