# Discrete Gaussian-process Bayesian optimization. No forecast/test data dependencies.
using LinearAlgebra, Statistics, Random, Distributions

scheme_domain(n, fractions, multipliers) = [(k=k, multiplier=m) for m in multipliers
    for k in unique(clamp.(round.(Int, n .* fractions), 1, n-1))]
scheme_point(c, domain) = [log(c.k)/log(maximum(x.k for x in domain)),
    log(c.multiplier)/log(maximum(x.multiplier for x in domain))]
function scheme_kernel(x, y, ell)
    r=sqrt(5.0)*norm(x.-y)/ell
    (1+r+r*r/3)*exp(-r)
end

"Fit a unit-amplitude Matérn-5/2 GP to standardized log losses by marginal likelihood."
function scheme_gp(history, domain)
    X=[scheme_point(h,domain) for h in history];y=log.([h.loss for h in history])
    center=mean(y);scale=max(std(y),1e-6);z=(y.-center)./scale
    best=nothing;bestll=-Inf
    for ell in (0.2,0.5,1.0,2.0), noise in (1e-6,1e-3,0.03)
        K=[scheme_kernel(x,y,ell) for x in X,y in X]+noise*I
        C=cholesky(Symmetric(K));alpha=C\z
        ll=-dot(z,alpha)/2-sum(log,diag(C.L))-length(z)*log(2pi)/2
        if ll>bestll
            bestll=ll;best=(X=X,C=C,alpha=alpha,ell=ell,noise=noise,center=center,scale=scale,z=z)
        end
    end
    best
end

function scheme_ei(model, c, domain)
    x=scheme_point(c,domain);v=[scheme_kernel(x,y,model.ell) for y in model.X]
    mu=dot(v,model.alpha);sigma=sqrt(max(0.0,1-dot(v,model.C\v)))
    delta=minimum(model.z)-mu
    sigma<1e-12 ? max(0.,delta) : delta*cdf(Normal(),delta/sigma)+sigma*pdf(Normal(),delta/sigma)
end

function scheme_next(history,domain,initial)
    seen=Set((h.k,h.multiplier) for h in history)
    pending=filter(c->!((c.k,c.multiplier) in seen),domain)
    isempty(pending) && error("Discrete search space exhausted")
    for c in initial
        if !((c.k,c.multiplier) in seen)
            c in domain || error("Initial proposal outside search domain")
            return (candidate=c,method="initial_design",ei=NaN,ell=NaN,noise=NaN)
        end
    end
    model=scheme_gp(history,domain);acquisition=[scheme_ei(model,c,domain) for c in pending]
    (candidate=pending[argmax(acquisition)],method="gp_expected_improvement",
        ei=maximum(acquisition),ell=model.ell,noise=model.noise)
end

"Fresh-trajectory blocks; all indices refer to fresh data, excluding the training prefix."
function scheme_blocks(tau, sync; reference_segments=320, gap_segments=20, validation_segments=40)
    reference=1:reference_segments*tau;cursor=last(reference)+gap_segments*tau
    origins=Int[]
    for _ in 1:2
        push!(origins,cursor+sync)
        cursor=last(origins)+validation_segments*tau+gap_segments*tau
    end
    (reference=reference,origins=origins,validation_length=validation_segments*tau,
        required=last(origins)+validation_segments*tau)
end

function scheme_select(history)
    feasible=filter(h->h.valid && isfinite(h.loss),history)
    isempty(feasible) && return nothing
    feasible[argmin(getproperty.(feasible,:loss))]
end
