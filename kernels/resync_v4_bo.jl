# New failure-aware search; legacy Pareto prototype and its numeric penalty are unchanged.
module ResyncV4BO
using LinearAlgebra, Random, Statistics, Distributions
export next_candidate, numeric, model_point, kernel
numeric(r)=r.valid && all(isfinite,(r.w1,r.spread_loss)) && min(r.w1,r.spread_loss)>=0
kernel(x,y,ell)=begin r=sqrt(5.)*norm(x.-y)/ell;(1+r+r*r/3)*exp(-r) end
model_point(c,domain)=[log1p(c.k)/log1p(maximum(x.k for x in domain)),log(c.multiplier)/log(maximum(x.multiplier for x in domain))]
function next_candidate(history,domain,initial;scales,seed=20261122,feasibility_ell=.5,explore_every=4)
    all(x->isfinite(x)&&x>0,scales)||error("Positive frozen scales required")
    bases=filter(r->r.k==0,history)
    length(bases)==1 && numeric(only(bases)) || error("A valid paired baseline is required")
    seen=Set((r.k,r.multiplier) for r in history)
    pending=filter(c->!((c.k,c.multiplier) in seen),domain)
    isempty(pending)&&error("Search exhausted")
    weight=rand(MersenneTwister(seed+length(history)),[.1,.3,.5,.7,.9])
    for c in initial
        c in domain||error("Initial point outside domain")
        if !((c.k,c.multiplier) in seen)
            return (candidate=c,method="initial_design",weight=weight,acquisition=NaN,feasibility=NaN)
        end
    end
    valid=filter(numeric,history);X=[model_point(r,domain) for r in valid]
    success(c)=begin
        ws=[kernel(model_point(r,domain),model_point(c,domain),feasibility_ell) for r in history]
        (1+sum(w*Int(numeric(r)) for (w,r) in zip(ws,history)))/(2+sum(ws))
    end
    if length(valid)<2
        scores=[minimum(norm(model_point(c,domain).-model_point(r,domain)) for r in history)*success(c) for c in pending]
        c=pending[argmax(scores)]
        return (candidate=c,method="distance_exploration_times_feasibility",weight=weight,acquisition=maximum(scores),feasibility=success(c))
    end
    ideal=(minimum(r.w1/scales[1] for r in valid),minimum(r.spread_loss/scales[2] for r in valid))
    scalar(r)=max(weight*(r.w1/scales[1]-ideal[1]),(1-weight)*(r.spread_loss/scales[2]-ideal[2]))+
        .05*(weight*r.w1/scales[1]+(1-weight)*r.spread_loss/scales[2])+1e-12
    y=log.(scalar.(valid));z=(y.-mean(y))./max(std(y),1e-6)
    best=nothing;bestll=-Inf
    for ell in [.2,.5,1.,2.],noise in [1e-6,1e-3,.03]
        K=[kernel(x,y,ell) for x in X,y in X]+noise*I;C=cholesky(Symmetric(K));alpha=C\z
        ll=-dot(z,alpha)/2-sum(log,diag(C.L))-length(z)*log(2pi)/2
        if ll>bestll;bestll=ll;best=(C=C,alpha=alpha,ell=ell);end
    end
    acquisition_step=count(r->r.k>0,history)-length(initial)+1
    exploration=acquisition_step%explore_every==0
    values=Float64[]
    for c in pending
        v=[kernel(x,model_point(c,domain),best.ell) for x in X]
        mu=dot(v,best.alpha);sigma=sqrt(max(0.,1-dot(v,best.C\v)));delta=minimum(z)-mu
        ei=sigma<1e-12 ? max(0.,delta) : delta*cdf(Normal(),delta/sigma)+sigma*pdf(Normal(),delta/sigma)
        push!(values,(exploration ? sigma : ei)*success(c))
    end
    c=pending[argmax(values)]
    (candidate=c,method=exploration ? "gp_uncertainty_times_feasibility" : "gp_ei_times_feasibility",
        weight=weight,acquisition=maximum(values),feasibility=success(c))
end
end
