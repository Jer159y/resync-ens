# Expanded, profile-controlled within-architecture Duffing ensemble study.
# No mixed ensemble. Earlier N10 outcomes are exploratory, not final evidence.
include("sequential_models.jl")
include("resync_v4_evaluator.jl")
using .ResyncV4Metrics,.ResyncV4Evaluator
const JOB=TOML.parsefile(ARGS[1])
const OUT2=JOB["output"]
const N2=Int(JOB["members"])
const TAU2=100
const SEGMENTS2=Int(JOB["segments"])
const TRAIN_UPDATES2=Int(JOB["updates"])
const FRACTION=Float64(JOB["fraction"])
@assert 0<FRACTION<1
# Standard absolute next-state readout, as in the main ESN study.
function step_abs(m,h,x)
    y,state,cache=step(m,h,x)
    y.-x[1:2],state,cache
end
function main_expanded()
    mkpath(OUT2);mkpath(JOB["model_root"])
    protocol=copy(JOB)
    protocol["source_sha256"]=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ["sequential_validation.jl","sequential_models.jl","resync_v4_metrics.jl","resync_v4_evaluator.jl"])
    protocol["inputs"]="q,v only; no forcing phase, fault injection, post-start truth or parameter updates"
    protocol["cohort"]="predeclared seeds; no member filtering/mixed ensemble; no 80percent initial-skill gate; shared models across fractions"
    protocol["interpretation"]="Full Resync vs no correction, own and returned records; not a boundary/history ablation or architecture ranking"
    path=joinpath(OUT2,"protocol.toml")
    if isfile(path);@assert TOML.parsefile(path)==protocol;else;save_toml(path,protocol);end
    fulltrain,_=trajectory(8000,2026112451);validation,_=trajectory(19000,2026112462)
    cal,_=trajectory(100000,2026112453)
    # All correction-fraction comparisons use validation only.
    Y=validation # validation only; no test trajectory is generated
    T=Int(JOB["training_samples"]);train=fulltrain[:,1:T]
    # Common normalization across profiles; all profiles contain this prefix.
    mu=mean(fulltrain[:,1:2000];dims=2);sd=std(fulltrain[:,1:2000];dims=2,corrected=false)
    train=(train.-mu)./sd;validation=(validation.-mu)./sd;cal=(cal.-mu)./sd;Y=(Y.-mu)./sd
    save_toml(joinpath(OUT2,"normalization.toml"),Dict("mean"=>vec(mu),"population_sd"=>vec(sd),"samples"=>T))
    target=cal[:,1:20000];pool=cal[:,22001:end];rng=MersenneTwister(2026112455);spreads=Float64[]
    starts=collect(1:TAU2:size(pool,2)-TAU2+1)
    @assert length(starts)>=N2
    for _ in 1:512
        ix=shuffle(rng,starts)[1:N2];push!(spreads,window_spread(cat([pool[:,i:i+TAU2-1] for i in ix]...;dims=3)))
    end
    sref=median(spreads);save_toml(joinpath(OUT2,"calibration.toml"),Dict("spread_reference"=>sref,"draws"=>512,"members"=>N2,"target_samples"=>20000))
    qs=collect(range(0.,1.;length=1001));q=quantile(vec(target[1,:]),qs)
    csvwrite(joinpath(OUT2,"reference_cdf.csv"),[(probability=qs[i],q=q[i]) for i in eachindex(qs)])
    csvwrite(joinpath(OUT2,"reference_phase.csv"),[(q=target[1,i],v=target[2,i]) for i in 1:10:size(target,2)])
    geometry=fit_feature2(train);models=ReplayToy[];cards=NamedTuple[];kind=Symbol(JOB["kind"]);n=Int(JOB["units"])
    for member in 1:N2
        # Identical initialization seeds across profiles whenever dimensions agree.
        seed=202611246+100000Int(JOB["bank"])+1000member+findfirst(==(kind),KINDS)
        p=joinpath(JOB["model_root"],"member_$member.bin");meta=p*".toml"
        if isfile(p)&&isfile(meta);m=deserialize(p);info=TOML.parsefile(meta);@assert fingerprint(m)==info["parameter_sha256"]
        else
            m=model2(kind,n,seed)
            kind==:esn && (m.p[:W].*=Float64(JOB["spectral_radius"])/.8)
            result=train2(m,train,validation,member+202611246+100000Int(JOB["bank"]))
            info=Dict("kind"=>string(kind),"member"=>member,"units"=>n,"parameters"=>sum(length,values(m.p)),"validation100_nrmse"=>result.bestscore,"selected_iteration"=>result.chosen,"trained_state_weight_change"=>result.change,"parameter_sha256"=>fingerprint(m))
            if kind==:ssm
                a=.995tanh.(m.p[:alpha]);info["driven_radius"]=maximum(abs,a);info["closed_loop_radius"]=maximum(abs,eigvals(Diagonal(a)+m.p[:B]*m.p[:V]))
            end
            serialize(p,m);save_toml(meta,info);csvwrite(p*"_training.csv",result.trace)
        end
        push!(models,m);push!(cards,(kind=string(kind),member=member,units=n,validation100_nrmse=info["validation100_nrmse"],trained_steps=info["selected_iteration"],state_weight_change=info["trained_state_weight_change"],driven_radius=get(info,"driven_radius",NaN),closed_loop_radius=get(info,"closed_loop_radius",NaN)))
        save_toml(joinpath(OUT2,"status.toml"),Dict("stage"=>"training","member"=>member,"members"=>N2));println("TRAIN $(JOB["stage"]) $(JOB["profile"]) $kind $member/$N2");flush(stdout)
    end
    csvwrite(joinpath(OUT2,"model_cards.csv"),cards)
    for origin in Int.(JOB["origins"])
        initial_path=joinpath(JOB["baseline_root"],"initial_skill_origin$origin.csv")
        if !isfile(initial_path)
            mkpath(dirname(initial_path));skills=NamedTuple[]
            for (i,m) in enumerate(models)
                h=drive2(m,zero_state(m),Y[:,origin-1000:origin-1]);P,_=rollout2(m,h,Y[:,origin],20)
                short=sqrt(mean(abs2,P.-Y[:,origin+1:origin+20]));h=drive2(m,zero_state(m),Y[:,origin-256:origin-1]);err=Float64[]
                for t in origin:origin+99
                    pred,h,_=step_abs(m,h,Y[:,t]);push!(err,mean(abs2,pred.-Y[:,t+1]))
                end
                teacher=sqrt(mean(err));push!(skills,(member=i,short20_nrmse=short,teacher_nrmse=teacher,initially_skilled=short<=.3&&teacher<=.05,truth_used_for_diagnostic_only=true))
            end
            csvwrite(initial_path,skills)
        end
        a=run_policy2(string(kind),models,Y,origin,target,sref,geometry;corrected=false)
        b=run_policy2(string(kind),models,Y,origin,target,sref,geometry;corrected=true)
        if a["status"]==b["status"]=="complete";@assert a["first_own_sha256"]==b["first_own_sha256"];end
        println("ROLLOUT $kind origin=$origin baseline=$(a["status"]) resync=$(b["status"])");flush(stdout)
    end
    save_toml(joinpath(OUT2,"complete.toml"),Dict("status"=>"complete","kind"=>string(kind),"members"=>N2,"origins"=>JOB["origins"],"profile"=>JOB["profile"],"parameters_fixed"=>true))
end

# Same tested BPTT equations, adjusted from four inputs to the two observed states.
source2=read(joinpath(@__DIR__,"sequential_models.jl"),String)
begin2=findfirst("function loss_grad(",source2).start;end2=findfirst("function csvwrite(",source2).start
gradient2=replace(source2[begin2:prevind(source2,end2)],"function loss_grad("=>"function loss_grad2(","[5:end]"=>"[3:end]","zeros(n+4)"=>"zeros(n+2)","pred[:,t],h,c=step("=>"pred[:,t],h,c=step_abs(")
include_string(@__MODULE__,gradient2,"two_observation_BPTT.jl")
function model2(kind,n,seed)
    m=make_model(kind,n,seed)
    if kind in (:rnn,:gru,:lstm)
        for k in (kind==:rnn ? [:W] : kind==:gru ? [:Z,:R,:G] : [:F,:I,:O,:G])
            m.p[k]=m.p[k][:,vcat(1:2,5:n+4)]
        end
    else
        m.p[:B]=m.p[:B][:,1:2]
    end
    m
end
function drive2(m,h,Y)
    h=copy(h)
    for i in axes(Y,2);_,h,_=step(m,h,Y[:,i]);end
    h
end
function rollout2(m,h,x,T)
    P=fill(NaN,2,T);h=copy(h);x=copy(x)
    for t in 1:T
        x,h,_=step_abs(m,h,x);P[:,t]=x
        if !all(isfinite,x)||!all(isfinite,h);break;end
    end
    P,h
end
function val2(m,Y)
    scores=Float64[]
    for o in [300,600,900,1200]
        h=drive2(m,zero_state(m),Y[:,o-256:o-1]);P=first(rollout2(m,h,Y[:,o],100))
        push!(scores,all(isfinite,P) ? sqrt(mean(abs2,P.-Y[:,o+1:o+100])) : Inf)
    end
    mean(scores)
end
function train2(m,Y,V,seed)
    rng=MersenneTwister(seed);before=deepcopy(m.p);trace=NamedTuple[];best=deepcopy(m.p);bestscore=Inf;chosen=0
    if m.kind==:esn
        h=zero_state(m);features=Vector{Vector{Float64}}();targets=Vector{Vector{Float64}}()
        for i in 1:size(Y,2)-1
            _,h,_=step(m,h,Y[:,i])
            if i>256;push!(features,vcat(h,1.));push!(targets,Y[:,i+1]);end
        end
        X=reduce(hcat,features);T=reduce(hcat,targets);coef=T*X'/(X*X'+Float64(get(JOB,"ridge",1e-5))*I)
        m.p[:V].=coef[:,1:m.n];m.p[:c].=coef[:,end];bestscore=val2(m,V);chosen=1
        @assert all(m.p[k]==before[k] for k in [:W,:B,:b])
        push!(trace,(iteration=1,training_loss=mean(abs2,coef*X.-T),validation100_nrmse=bestscore))
    else
        mom=Dict(k=>zero(v) for (k,v) in m.p);vel=deepcopy(mom)
        for it in 1:TRAIN_UPDATES2
            o=rand(rng,1:size(Y,2)-97);loss,g=loss_grad2(m,Y[:,o:o+95],Y[:,o+1:o+96])
            isfinite(loss)||error("Nonfinite training")
            gn=sqrt(sum(sum(abs2,v) for v in values(g)));clip=min(1.,1/max(gn,1e-12))
            for k in keys(m.p)
                grad=g[k].*clip;mom[k].=.9mom[k].+.1grad;vel[k].=.999vel[k].+.001grad.^2
                rate=it<=3000 ? .001 : .0003
                m.p[k].-=rate .* (mom[k]./(1-.9^it))./(sqrt.(vel[k]./(1-.999^it)).+1e-8)
            end
            if it==TRAIN_UPDATES2
                score=val2(m,V);push!(trace,(iteration=it,training_loss=loss,validation100_nrmse=score))
                if score<bestscore;bestscore=score;best=deepcopy(m.p);chosen=it;end
            end
        end
        chosen>0||error("No finite checkpoint")
        for k in keys(m.p);m.p[k].=best[k];end
    end
    sk=m.kind==:rnn ? [:W] : m.kind==:gru ? [:Z,:R,:G] : m.kind==:lstm ? [:F,:I,:O,:G] : m.kind==:ssm ? [:alpha,:B] : [:W,:B]
    change=sqrt(sum(sum(abs2,m.p[k].-before[k]) for k in sk))
    (;chosen,bestscore,change,trace)
end
function feature2(A)
    all(isfinite,A)||return fill(NaN,8)
    v=Float64[]
    for d in 1:2
        x=vec(A[d,:]);scale=max(maximum(abs,x),1.);z=x./scale;mu=mean(z);dev=z.-mu;sd=sqrt(mean(abs2,dev))
        append!(v,[scale*mu,scale*sd,sd>1e-14 ? mean(dev.^3)/sd^3 : 0.,sd>1e-14 ? mean(dev.^4)/sd^4-3 : 0.])
    end
    v
end
function fit_feature2(train)
    F=hcat([feature2(train[:,s:s+TAU2-1]) for s in 1:TAU2:size(train,2)-TAU2+1]...)
    mu=mean(F;dims=2);sd=max.(std(F;dims=2,corrected=false),1e-10);Z=(F.-mu)./sd;S=svd(Z)
    q=max(2,something(findfirst(>=(.95),cumsum(S.S.^2)./sum(abs2,S.S)),length(S.S)))
    (;mu=vec(mu),sd=vec(sd),P=S.U[:,1:q])
end
function select2(candidates,geometry)
    features=[feature2(A) for A in candidates];finite=[all(isfinite,v) for v in features]
    eligible=findall(finite);scores=fill(Inf,length(candidates))
    if !isempty(eligible)
        Z=hcat([geometry.P'*((features[i].-geometry.mu)./geometry.sd) for i in eligible]...);center=mean(Z;dims=2)
        for (j,i) in enumerate(eligible);scores[i]=norm(Z[:,j].-vec(center));end
    end
    # Correction fraction varied on validation with identical models/baseline.
    bad=sortperm(scores;rev=true)[1:max(1,round(Int,FRACTION*length(candidates)))];good=[i for i in eligible if !(i in bad)]
    bad,good,scores
end
scalarrows2(rows)=[(; (k=>v for (k,v) in pairs(r) if !(v isa AbstractArray))...) for r in rows]
function run_policy2(label,models,Y,origin,target,sref,geometry;corrected)
    out=corrected ? joinpath(OUT2,"cases",label,"origin$origin","resync") : joinpath(JOB["baseline_root"],"cases",label,"origin$origin","no_correction");done=joinpath(out,"complete.toml")
    isfile(done)&&return TOML.parsefile(done)
    mkpath(out);N=length(models);saved=[drive2(m,zero_state(m),Y[:,origin-1000:origin-1]) for m in models];states=deepcopy(saved);x0=Y[:,origin];inputs=[copy(x0) for _ in 1:N]
    histories=[zeros(2,0) for _ in 1:N];hashes=fingerprint.(models);own=WindowEvaluator(TAU2,N,SEGMENTS2,target,sref);effective=WindowEvaluator(TAU2,N,SEGMENTS2,target,sref)
    events=NamedTuple[];diagnostics=NamedTuple[];firsthash="";started=time();qown=Float64[];qeff=Float64[];phaseown=NamedTuple[];phaseeff=NamedTuple[]
    for segment in 1:SEGMENTS2
        candidates=Matrix{Float64}[];nextstates=Vector{Float64}[]
        for i in 1:N
            A,h=rollout2(models[i],states[i],inputs[i],TAU2);push!(candidates,A);push!(nextstates,h)
        end
        A=cat(candidates...;dims=3);B=copy(A)
        segment==1 && (firsthash=bytes2hex(sha256(reinterpret(UInt8,vec(A)))))
        states=nextstates;inputs=[copy(C[:,end]) for C in candidates]
        if corrected
            for i in 1:N;histories[i]=hcat(histories[i],candidates[i]);end
            bad,good,scores=select2(candidates,geometry)
            if isempty(good)
                save_toml(done,Dict("status"=>"no_finite_donor","completed_segments"=>segment-1,"parameters_unchanged"=>fingerprint.(models)==hashes));return TOML.parsefile(done)
            end
            for (j,i) in enumerate(bad)
                donor=good[mod1(j,length(good))];history=copy(histories[donor]);histories[i]=history;B[:,:,i].=candidates[donor]
                # Full generated history in observation space; recipient weights/state coordinates are its own.
                replayinputs=hcat(x0,history[:,1:end-1]);states[i]=drive2(models[i],saved[i],replayinputs);inputs[i]=copy(history[:,end])
                evaltruth=Y[:,origin+(segment-1)*TAU2+1:origin+segment*TAU2]
                push!(events,(segment=segment,recipient=i,recipient_kind=string(models[i].kind),recipient_units=models[i].n,donor=donor,donor_kind=string(models[donor].kind),donor_units=models[donor].n,replayed_steps=size(replayinputs,2),score=scores[i],recipient_rmse=sqrt(mean(abs2,candidates[i].-evaltruth)),donor_rmse=sqrt(mean(abs2,candidates[donor].-evaltruth)),truth_for_evaluation_only=true))
            end
        end
        range=(segment-1)*TAU2+1:segment*TAU2;y=Y[:,origin.+range];push_base!(own,A,y);push_base!(effective,B,y)
        for (record,Q) in [("own",A),("returned",B)]
            accuracy=Float64[];diversity=Float64[]
            for d in 1:2,t in 1:5:TAU2
                xs=vec(Q[d,t,:])
                if all(isfinite,xs)
                    ac=mean(abs.(xs.-y[d,t]));push!(accuracy,ac);push!(diversity,ac-empirical_crps(xs,y[d,t]))
                else
                    push!(accuracy,NaN);push!(diversity,NaN)
                end
            end
            push!(diagnostics,(segment=segment,record=record,crps_accuracy_term=mean(accuracy),crps_diversity_term=mean(diversity),n_nonfinite=count(i->!all(isfinite,Q[:,:,i]),1:N),n_low_variance=count(i->all(isfinite,Q[:,:,i])&&sqrt(mean(var(Q[:,:,i];dims=2,corrected=false)))<1e-4,1:N),n_amplitude_gt10=count(i->any(x->isfinite(x)&&abs(x)>10,Q[:,:,i]),1:N)))
        end
        if segment>SEGMENTS2-10
            append!(qown,vec(A[1,:,:]));append!(qeff,vec(B[1,:,:]))
            for t in 1:10:TAU2,i in 1:N
                push!(phaseown,(segment=segment,member=i,q=A[1,t,i],v=A[2,t,i]));push!(phaseeff,(segment=segment,member=i,q=B[1,t,i],v=B[2,t,i]))
            end
        end
        @assert fingerprint.(models)==hashes
        save_toml(joinpath(OUT2,"status.toml"),Dict("stage"=>"rollout","case"=>label,"origin"=>origin,"policy"=>corrected ? "resync" : "no_correction","segment"=>segment,"segments"=>SEGMENTS2))
    end
    a=finish(own);b=finish(effective)
    for (scope,e) in [("own",own),("effective",effective)]
        csvwrite(joinpath(out,scope*"_base_windows.csv"),scalarrows2(e.base_rows));csvwrite(joinpath(out,scope*"_distribution_windows.csv"),e.distribution_rows)
    end
    csvwrite(joinpath(out,"own_summary.csv"),[a]);csvwrite(joinpath(out,"effective_summary.csv"),[b])
    !isempty(events)&&csvwrite(joinpath(out,"events.csv"),events)
    csvwrite(joinpath(out,"quality_diagnostics.csv"),diagnostics)
    for (scope,q,phase) in [("own",qown,phaseown),("effective",qeff,phaseeff)]
        quantiles=collect(range(0.,1.;length=1001));values=all(isfinite,q) ? quantile(q,quantiles) : fill(NaN,1001)
        csvwrite(joinpath(out,scope*"_cdf.csv"),[(probability=quantiles[i],q=values[i]) for i in eachindex(quantiles)])
        csvwrite(joinpath(out,scope*"_phase.csv"),phase)
    end
    save_toml(done,Dict("status"=>"complete","valid"=>b.valid,"own_valid"=>a.valid,"members"=>N,"segments"=>SEGMENTS2,"first_own_sha256"=>firsthash,"parameters_unchanged"=>fingerprint.(models)==hashes,"seconds"=>time()-started,"current_record_copy"=>corrected,"history_replay"=>corrected,"donor_boundary_input"=>corrected,"truth_used_for_selection_or_replay"=>false))
    TOML.parsefile(done)
end

function check2()
    @testset "Two-observation model gradients and history alignment" begin
        for kind in (:rnn,:gru,:lstm,:ssm)
            m=model2(kind,3,42);rng=MersenneTwister(5);X=.2randn(rng,2,8);Y=.2randn(rng,2,8);_,g=loss_grad2(m,X,Y;warmup=1)
            for k in keys(m.p),i in unique([first(eachindex(m.p[k])),last(eachindex(m.p[k]))])
                a=m.p[k];v=a[i];eps=1e-6;a[i]=v+eps;plus=first(loss_grad2(m,X,Y;warmup=1));a[i]=v-eps;minus=first(loss_grad2(m,X,Y;warmup=1));a[i]=v
                @test isapprox(g[k][i],(plus-minus)/(2eps);rtol=3e-4,atol=2e-6)
            end
        end
        for kind in KINDS
            m=model2(kind,5,9);h=zero_state(m);x=[.1,.2];P,hend=rollout2(m,h,x,13);reconstructed=drive2(m,h,hcat(x,P[:,1:12]))
            @test hend==reconstructed
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    check2()
    "--test-only" in ARGS || main_expanded()
end
