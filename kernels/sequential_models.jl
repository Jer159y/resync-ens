# Small laptop feasibility experiment; not an architecture-ranking benchmark.
# All training and state-update equations are explicit; no new package installs.
using LinearAlgebra,Random,Statistics,SHA,TOML,Serialization,Test
BLAS.set_num_threads(1)
const ROOT="outputs/research/duffing_heterogeneous_replay_20261124"
const KINDS=(:rnn,:gru,:lstm,:ssm,:esn)
struct ReplayToy
    kind::Symbol
    n::Int
    p::Dict{Symbol,Array{Float64}}
end
sig(x)=1/(1+exp(-clamp(x,-50.,50.)))
zero_state(m)=zeros(m.kind==:lstm ? 2m.n : m.n)
fingerprint(m)=bytes2hex(sha256(reduce(vcat,[reinterpret(UInt8,vec(m.p[k])) for k in sort(collect(keys(m.p)))])))
function make_model(kind,n,seed)
    rng=MersenneTwister(seed);p=Dict{Symbol,Array{Float64}}(:V=>.01randn(rng,2,n),:c=>zeros(2))
    if kind==:rnn
        p[:W]=.4randn(rng,n,n+4)/sqrt(n);p[:W][:,1:4].*=sqrt(n);p[:b]=zeros(n)
    elseif kind in (:gru,:lstm)
        pairs=kind==:gru ? ((:Z,:bz),(:R,:br),(:G,:bg)) : ((:F,:bf),(:I,:bi),(:O,:bo),(:G,:bg))
        for (w,b) in pairs
            p[w]=.3randn(rng,n,n+4)/sqrt(n);p[w][:,1:4].*=sqrt(n);p[b]=zeros(n)
        end
        kind==:lstm && (p[:bf].=1.)
    elseif kind==:ssm
        p[:alpha]=atanh.(collect(range(.2,.99;length=n))./.995);p[:B]=.1randn(rng,n,4)
    else
        p[:W]=randn(rng,n,n);p[:W].*=.8/maximum(abs,eigvals(p[:W]));p[:B]=.3randn(rng,n,4);p[:b]=.1randn(rng,n)
    end
    ReplayToy(kind,n,p)
end
function step(m,state,x)
    p=m.p;n=m.n;h=state[1:n];q=vcat(x,h)
    if m.kind==:rnn
        hn=tanh.(p[:W]*q.+p[:b]);cache=(q=q,h=hn,)
    elseif m.kind==:gru
        z=sig.(p[:Z]*q.+p[:bz]);r=sig.(p[:R]*q.+p[:br]);qg=vcat(x,r.*h)
        g=tanh.(p[:G]*qg.+p[:bg]);hn=(1 .-z).*h.+z.*g
        cache=(q=q,qg=qg,hprev=h,h=hn,z=z,r=r,g=g)
    elseif m.kind==:lstm
        c=state[n+1:2n];f=sig.(p[:F]*q.+p[:bf]);i=sig.(p[:I]*q.+p[:bi]);o=sig.(p[:O]*q.+p[:bo]);g=tanh.(p[:G]*q.+p[:bg])
        cn=f.*c.+i.*g;hn=o.*tanh.(cn);cache=(q=q,h=hn,cprev=c,c=cn,f=f,i=i,o=o,g=g)
        return x[1:2].+p[:V]*hn.+p[:c],vcat(hn,cn),cache
    elseif m.kind==:ssm
        a=.995tanh.(p[:alpha]);hn=a.*h.+p[:B]*x;cache=(h=hn,hprev=h,a=a,x=x)
    else
        hn=.5h.+.5tanh.(p[:W]*h.+p[:B]*x.+p[:b]);cache=(h=hn,)
    end
    x[1:2].+p[:V]*hn.+p[:c],hn,cache
end
function loss_grad(m,X,Y;warmup=16)
    m.kind==:esn && error("ESN readout uses ridge")
    p=m.p;n=m.n;T=size(X,2);g=Dict(k=>zero(v) for (k,v) in p);caches=Any[];pred=zeros(2,T);h=zero_state(m)
    for t in 1:T;pred[:,t],h,c=step(m,h,X[:,t]);push!(caches,c);end
    loss=mean(abs2,pred[:,warmup+1:end].-Y[:,warmup+1:end]);dhnext=zeros(n);dcnext=zeros(n)
    for t in T:-1:1
        c=caches[t];dy=t>warmup ? (pred[:,t].-Y[:,t])./(T-warmup) : zeros(2)
        g[:V].+=dy*c.h';g[:c].+=dy;dh=p[:V]'*dy.+dhnext
        if m.kind==:rnn
            da=dh.*(1 .-c.h.^2);g[:W].+=da*c.q';g[:b].+=da;dhnext=(p[:W]'*da)[5:end]
        elseif m.kind==:gru
            dg=dh.*c.z;dz=dh.*(c.g.-c.hprev);dhprev=dh.*(1 .-c.z);dag=dg.*(1 .-c.g.^2)
            g[:G].+=dag*c.qg';g[:bg].+=dag;dqg=p[:G]'*dag;dr=dqg[5:end].*c.hprev;dhprev.+=dqg[5:end].*c.r
            dar=dr.*c.r.*(1 .-c.r);daz=dz.*c.z.*(1 .-c.z)
            g[:R].+=dar*c.q';g[:br].+=dar;g[:Z].+=daz*c.q';g[:bz].+=daz
            dhnext=dhprev.+(p[:R]'*dar.+p[:Z]'*daz)[5:end]
        elseif m.kind==:lstm
            tc=tanh.(c.c);dc=dcnext.+dh.*c.o.*(1 .-tc.^2)
            gg=((:F,:bf,dc.*c.cprev.*c.f.*(1 .-c.f)),(:I,:bi,dc.*c.g.*c.i.*(1 .-c.i)),(:O,:bo,dh.*tc.*c.o.*(1 .-c.o)),(:G,:bg,dc.*c.i.*(1 .-c.g.^2)))
            dq=zeros(n+4)
            for (w,b,da) in gg;g[w].+=da*c.q';g[b].+=da;dq.+=p[w]'*da;end
            dhnext=dq[5:end];dcnext=dc.*c.f
        else
            g[:alpha].+=dh.*c.hprev.*(.995 .* (1 .-(c.a./.995).^2));g[:B].+=dh*c.x';dhnext=dh.*c.a
        end
    end
    loss,g
end
function csvwrite(path,rows)
    mkpath(dirname(path));open(path,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for r in rows;println(io,join(string.(values(r)),','));end
    end
end
function save_toml(path,obj)
    mkpath(dirname(path));open(io->TOML.print(io,obj;sorted=true),path,"w")
end
function rhs(x,t)
    [x[2],x[1]-x[1]^3-.3x[2]+.4cos(1.2t)]
end
function trajectory(N,seed;dt=.1,h=.01)
    rng=MersenneTwister(seed);x=.2randn(rng,2);t=2pi*rand(rng)/1.2;sub=round(Int,dt/h)
    function advance(x,t)
        a=rhs(x,t);b=rhs(x.+h/2 .*a,t+h/2);c=rhs(x.+h/2 .*b,t+h/2);d=rhs(x.+h.*c,t+h)
        x.+h/6 .*(a.+2b.+2c.+d)
    end
    for _ in 1:round(Int,500/h);x=advance(x,t);t+=h;end
    Y=zeros(2,N);times=zeros(N)
    for i in 1:N
        for _ in 1:sub;x=advance(x,t);t+=h;end
        Y[:,i]=x;times[i]=t
    end
    Y,times
end
function input(y,t)
    vcat(y,cos(1.2t),sin(1.2t))
end
function drive(m,h,Y,times)
    h=copy(h)
    for i in axes(Y,2);_,h,_=step(m,h,input(Y[:,i],times[i]));end
    h
end
function rollout(m,h,y,times)
    pred=fill(NaN,2,length(times));h=copy(h);y=copy(y)
    for i in eachindex(times)
        y,h,_=step(m,h,input(y,times[i]));pred[:,i]=y
        if !all(isfinite,y) || !all(isfinite,h) || maximum(abs,y)>1e6;return pred,h,y;end
    end
    pred,h,y
end
function validation_score(m,Y,times)
    scores=Float64[]
    for o in [300,600,900,1200]
        h=drive(m,zero_state(m),Y[:,o-256:o-1],times[o-256:o-1])
        P=first(rollout(m,h,Y[:,o],times[o:o+19]));push!(scores,all(isfinite,P) ? sqrt(mean(abs2,P.-Y[:,o+1:o+20])) : Inf)
    end
    mean(scores)
end
function train_model(m,Y,times,V,vt,seed;updates=2000)
    rng=MersenneTwister(seed);trace=NamedTuple[];before=deepcopy(m.p);mom=Dict(k=>zero(v) for (k,v) in m.p);vel=deepcopy(mom)
    best=deepcopy(m.p);bestscore=Inf;chosen=0
    if m.kind==:esn
        h=zero_state(m);features=Vector{Vector{Float64}}();targets=Vector{Vector{Float64}}()
        for i in 1:size(Y,2)-1
            _,h,_=step(m,h,input(Y[:,i],times[i]))
            if i>256;push!(features,vcat(h,1.));push!(targets,Y[:,i+1].-Y[:,i]);end
        end
        X=reduce(hcat,features);T=reduce(hcat,targets);coef=T*X'/(X*X'+1e-5I)
        m.p[:V].=coef[:,1:m.n];m.p[:c].=coef[:,end];bestscore=validation_score(m,V,vt);chosen=1
        @assert m.p[:W]==before[:W] && m.p[:B]==before[:B] && m.p[:b]==before[:b]
        push!(trace,(iteration=1,training_loss=mean(abs2,coef*X.-T),validation20_nrmse=bestscore))
    else
        for it in 1:updates
            o=rand(rng,1:size(Y,2)-97);indices=o:o+95
            X=reduce(hcat,[input(Y[:,i],times[i]) for i in indices]);T=Y[:,o+1:o+96]
            loss,g=loss_grad(m,X,T);isfinite(loss) || error("Nonfinite training loss")
            normg=sqrt(sum(sum(abs2,v) for v in values(g)));clip=min(1.,1/max(normg,1e-12))
            for k in keys(m.p)
                grad=g[k].*clip;mom[k].=.9mom[k].+.1grad;vel[k].=.999vel[k].+.001grad.^2
                m.p[k].-=.001 .* (mom[k]./(1-.9^it))./(sqrt.(vel[k]./(1-.999^it)).+1e-8)
            end
            if it in [500,1000,2000]
                score=validation_score(m,V,vt);push!(trace,(iteration=it,training_loss=loss,validation20_nrmse=score))
                if score<bestscore;bestscore=score;best=deepcopy(m.p);chosen=it;end
            end
        end
        chosen>0 || error("No finite validation checkpoint: $(m.kind)")
        for k in keys(m.p);m.p[k].=best[k];end
    end
    statekeys=m.kind==:rnn ? [:W] : m.kind==:gru ? [:Z,:R,:G] : m.kind==:lstm ? [:F,:I,:O,:G] : m.kind==:ssm ? [:alpha,:B] : [:W,:B]
    change=sqrt(sum(sum(abs2,m.p[k].-before[k]) for k in statekeys))
    (;chosen,bestscore,change,trace)
end
function tests()
    @testset "Multivariate residual predictor gradients and SSM replay identity" begin
        checks=0
        for kind in (:rnn,:gru,:lstm,:ssm)
            m=make_model(kind,3,42);rng=MersenneTwister(4);X=.2randn(rng,4,8);Y=.2randn(rng,2,8);_,g=loss_grad(m,X,Y;warmup=1)
            for k in keys(m.p),i in unique([first(eachindex(m.p[k])),last(eachindex(m.p[k]))])
                a=m.p[k];old=a[i];eps=1e-6;a[i]=old+eps;plus=first(loss_grad(m,X,Y;warmup=1));a[i]=old-eps;minus=first(loss_grad(m,X,Y;warmup=1));a[i]=old
                @test isapprox(g[k][i],(plus-minus)/(2eps);rtol=3e-4,atol=2e-6);checks+=1
            end
        end
        m=make_model(:ssm,5,7);rng=MersenneTwister(8);a=randn(rng,5);b=randn(rng,5);d0=a-b
        for _ in 1:17;x=randn(rng,4);_,a,_=step(m,a,x);_,b,_=step(m,b,x);end
        @test isapprox(a-b,(.995tanh.(m.p[:alpha])).^17 .*d0;atol=1e-12)
        println("Gradient checks: $checks; analytic shared-input SSM identity checked")
    end
end
function run()
    mkpath(ROOT);mkpath(joinpath(ROOT,"models"))
    protocol=Dict("version"=>"duffing_heterogeneous_replay_v1","source_sha256"=>bytes2hex(open(sha256,@__FILE__)),"device"=>"laptop_CPU_BLAS1",
      "physics"=>"q'=v; v'=q-q^3-0.3v+0.4cos(1.2t); RK4h0.01,save0.1,transient500",
      "training_seed"=>2411,"validation_seed"=>2412,"test_seed"=>2413,"training_samples"=>8000,"validation_samples"=>1500,"test_samples"=>3000,
      "known_forcing"=>"cos/sin phase supplied at every step; this is prescribed forcing, not poststart truth",
      "normalization"=>"training coordinate mean/population_sd; shared physical observation channel q,v",
      "models"=>["vanilla tanh RNN","Cho-style GRU reset before candidate recurrent map","standard LSTM hidden+cell","stable real diagonal linear SSM, not S4/Mamba","leaky tanh ESN"],
      "hidden_units_by_seed"=>[16,24,32],"esn_units_by_seed"=>[32,48,64],"seeds"=>[1,2,3],"readout"=>"linear residual: next qv=current qv+Vh+c",
      "training"=>"single-layer causal; neural/SSM all weights BPTT96,warm16,Adam0.001,clip1,2000updates; best validation20 checkpoint among500/1000/2000; ESN recurrence/input fixed,ridge1e-5",
      "test_origins"=>[600,1200,1800,2400],"warmup"=>256,"event_step"=>64,"horizon"=>40,"fault_scales"=>[0.,.5],
      "fault"=>"at initialized state, scale times max(state_RMS,0.1) times standard normal; LSTM perturbs h and c",
      "donors"=>"all five architecture types; donor seed is recipientseed mod3+1, giving different size/weights; same observed initialization and forcing",
      "factors"=>["donor boundary input","full generated-history state replay"],"copy"=>"no output-record copy; all scores next own forecasts",
      "mismatched_control"=>"replayed state with old-own next input is deliberately inconsistent",
      "scope"=>"exploratory feasibility; no architecture ranking, ensemble-health guarantee or optimal hyperparameter claim")
    protocolpath=joinpath(ROOT,"protocol.toml")
    if isfile(protocolpath);@assert TOML.parsefile(protocolpath)==protocol;else;save_toml(protocolpath,protocol);end
    train,tt=trajectory(8000,2411);val,vt=trajectory(1500,2412);test,st=trajectory(3000,2413)
    mu=mean(train;dims=2);sd=std(train;dims=2,corrected=false);train=(train.-mu)./sd;val=(val.-mu)./sd;test=(test.-mu)./sd
    save_toml(joinpath(ROOT,"normalization.toml"),Dict("mean"=>vec(mu),"population_sd"=>vec(sd)))
    models=Dict{Tuple{Symbol,Int},ReplayToy}();training=NamedTuple[]
    for kind in KINDS,seed in 1:3
        n=(kind==:esn ? [32,48,64] : [16,24,32])[seed];path=joinpath(ROOT,"models","$(kind)_seed$seed.bin")
        info=joinpath(ROOT,"models","$(kind)_seed$seed.toml")
        if isfile(path)&&isfile(info)
            m=deserialize(path);meta=TOML.parsefile(info)
        else
            m=make_model(kind,n,100seed+findfirst(==(kind),KINDS));result=train_model(m,train,tt,val,vt,seed+300)
            meta=Dict("kind"=>string(kind),"seed"=>seed,"hidden_units"=>n,"state_dimension"=>length(zero_state(m)),"parameter_count"=>sum(length,values(m.p)),"selected_iteration"=>result.chosen,"validation20_nrmse"=>result.bestscore,"trained_state_weight_change"=>result.change,"parameter_sha256"=>fingerprint(m))
            serialize(path,m);save_toml(info,meta);csvwrite(joinpath(ROOT,"$(kind)_seed$(seed)_training.csv"),result.trace)
        end
        models[(kind,seed)]=m;push!(training,(kind=string(kind),seed=seed,hidden_units=n,state_dimension=meta["state_dimension"],parameter_count=meta["parameter_count"],selected_iteration=meta["selected_iteration"],validation20_nrmse=meta["validation20_nrmse"],state_weight_change=meta["trained_state_weight_change"]))
        println("TRAINED $kind seed=$seed n=$n val20=$(meta["validation20_nrmse"])");flush(stdout)
        save_toml(joinpath(ROOT,"status.toml"),Dict("stage"=>"training","kind"=>string(kind),"seed"=>seed))
    end
    csvwrite(joinpath(ROOT,"training_summary.csv"),training)
    rows=NamedTuple[];curves=NamedTuple[];event=64;H=40;hashes=Dict(k=>fingerprint(m) for (k,m) in models)
    for kind in KINDS,seed in 1:3
        recipient=models[(kind,seed)]
        for (oi,o) in enumerate([600,1200,1800,2400]),faultscale in [0.,.5]
            saved=drive(recipient,zero_state(recipient),test[:,o-256:o-1],st[o-256:o-1]);fault=saved.+faultscale*max(sqrt(mean(abs2,saved)),.1).*randn(MersenneTwister(1000seed+oi),length(saved))
            own,endpoint,lastinput=rollout(recipient,fault,test[:,o],st[o:o+event-1])
            for donor_kind in KINDS
                donor_seed=seed%3+1;donor=models[(donor_kind,donor_seed)];dh=drive(donor,zero_state(donor),test[:,o-256:o-1],st[o-256:o-1])
                history,_,_=rollout(donor,dh,test[:,o],st[o:o+event-1]);replayhistory=hcat(test[:,o],history[:,1:end-1])
                rh=drive(recipient,saved,replayhistory,st[o:o+event-1]);truth=test[:,o+event+1:o+event+H]
                for b in 0:1,r in 0:1
                    state=r==1 ? rh : endpoint;xin=b==1 ? history[:,end] : lastinput
                    future=first(rollout(recipient,state,xin,st[o+event:o+event+H-1]));valid=all(isfinite,future)
                    e20=valid ? sqrt(mean(abs2,future[:,1:20].-truth[:,1:20])) : NaN;e40=valid ? sqrt(mean(abs2,future.-truth)) : NaN
                    push!(rows,(recipient=string(kind),recipient_seed=seed,recipient_units=recipient.n,donor=string(donor_kind),donor_seed=donor_seed,donor_units=donor.n,origin=oi,fault_scale=faultscale,boundary_input=b,history_replay=r,valid=valid,next20_nrmse=e20,next40_nrmse=e40,parameters_unchanged=fingerprint(recipient)==hashes[(kind,seed)]))
                    for h in 1:H
                        push!(curves,(recipient=string(kind),recipient_seed=seed,donor=string(donor_kind),origin=oi,fault_scale=faultscale,boundary_input=b,history_replay=r,lead=h,normalized_error=all(isfinite,future[:,h]) ? sqrt(mean(abs2,future[:,h].-truth[:,h])) : NaN))
                    end
                end
            end
        end
        println("EVALUATED $kind seed=$seed");flush(stdout)
    end
    @assert length(rows)==2400 && all(r->r.parameters_unchanged,rows)
    csvwrite(joinpath(ROOT,"conditions.csv"),rows);csvwrite(joinpath(ROOT,"lead_errors.csv"),curves)
    save_toml(joinpath(ROOT,"complete.toml"),Dict("status"=>"complete","conditions"=>length(rows),"models"=>length(models),"parameters_unchanged"=>all(fingerprint(m)==hashes[k] for (k,m) in models),"finite_conditions"=>count(r->r.valid,rows),"test_used_for_checkpoint_selection"=>false,"source_sha256"=>bytes2hex(open(sha256,@__FILE__))))
    save_toml(joinpath(ROOT,"status.toml"),Dict("stage"=>"complete"))
    println("DUFFING_HETEROGENEOUS_REPLAY_COMPLETE")
end
if abspath(PROGRAM_FILE)==@__FILE__
    tests();"--test-only" in ARGS || run()
end
