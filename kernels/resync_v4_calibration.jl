if !isdefined(@__MODULE__,:ResyncV4Metrics)
    include(joinpath(@__DIR__,"resync_v4_metrics.jl"))
end
module ResyncV4Calibration
using Random, Statistics
using ..ResyncV4Metrics
export calibrate, distribution_reference

function ensemble_window(R,starts,L)
    A=Array{Float64,3}(undef,size(R,1),L,length(starts))
    for (m,s) in enumerate(starts);A[:,:,m].=view(R,:,s:(s+L-1));end
    A
end

function sorted_w1(a,b)
    i=j=1;left=total=0.;n=length(a);m=length(b)
    while i<=n && j<=m
        ca=i/n;cb=j/m;right=min(ca,cb)
        total+=(right-left)*abs(a[i]-b[j]);left=right
        ca<=right && (i+=1);cb<=right && (j+=1)
    end
    total
end

"Exact pooled W1 references: sort the population pool once per coordinate, then filter whole-window draws."
function distribution_reference(target,pool,N,L;draws=128,seed=20261127,progress=false)
    rng=MersenneTwister(seed);T=size(pool,2)
    T>=N*L || return (status="unavailable",values=Float64[],reason="insufficient_nonoverlapping_windows")
    size(target,1)==size(pool,1) && all(isfinite,target) && all(isfinite,pool) || error("Invalid reference coordinates")
    masks=BitVector[]
    for _ in 1:draws
        starts=reference_starts(rng,T,L,N);mask=falses(T)
        for s in starts;mask[s:(s+L-1)].=true;end
        push!(masks,mask)
    end
    values=zeros(draws);D=size(pool,1)
    for d in 1:D
        order=sortperm(view(pool,d,:));b=sort(collect(view(target,d,:)))
        for rep in 1:draws
            mask=masks[rep]
            a=Float64[pool[d,t] for t in order if mask[t]]
            values[rep]+=sorted_w1(a,b)/D
        end
        if progress
            println("CAL W1 coordinate $d/$D");flush(stdout)
        end
    end
    (;status="computed",values,reason="")
end

"Fit and check draws are separate; check truths come from a disjoint bank. Inputs are standardized."
function calibrate(target,pool,checks,base,N;fit_draws=512,check_draws=512,
        distribution_draws=128,seed=20261125,dimension_samples=64,constant_tol=1e-10,
        numerical_floor=1e-12,progress=false)
    size(target,1)==size(pool,1)==size(checks,1) || error("Calibration coordinate mismatch")
    all(isfinite,target) && all(isfinite,pool) && all(isfinite,checks) || error("Nonfinite calibration data")
    base>=2 && N>=2 && min(fit_draws,check_draws,distribution_draws)>=2 || error("Invalid calibration design")
    numerical_floor>0 || error("Positive numerical floor required")
    size(pool,2)>=N*base && size(checks,2)>=base || error("Insufficient base calibration data")
    rng=MersenneTwister(seed);spreads=Float64[];dimensions=Float64[]
    for rep in 1:fit_draws
        starts=reference_starts(rng,size(pool,2),base,N)
        A=ensemble_window(pool,starts,base)
        push!(spreads,window_spread(A))
        push!(dimensions,trajectory_dimension(A;samples=dimension_samples,constant_tol).effective_dimension)
        if progress && rep%32==0
            println("CAL base fit $rep/$fit_draws");flush(stdout)
        end
    end
    sref=median(spreads)
    isfinite(sref) && sref>0 || error("Invalid fitted spread reference")
    spread_losses=Float64[];cover50=Float64[];cover90=Float64[];covdev=Float64[]
    for rep in 1:check_draws
        starts=reference_starts(rng,size(pool,2),base,N)
        A=ensemble_window(pool,starts,base)
        s=rand(rng,1:(size(checks,2)-base+1))
        truth=view(checks,:,s:(s+base-1))
        p=probability_metrics(A,truth;stride=max(1,base÷20))
        push!(spread_losses,abs(log(window_spread(A)/sref)))
        push!(cover50,p.coverage50);push!(cover90,p.coverage90);push!(covdev,p.coverage90_absdev)
        if progress && rep%32==0
            println("CAL base check $rep/$check_draws");flush(stdout)
        end
    end
    w=distribution_reference(target,pool,N,10base;draws=distribution_draws,seed=seed+2,progress)
    # Means of per-window reference losses give scales in the same units as the
    # equal-window validation mean. No independence or unbiased-population claim.
    floor_s=max(numerical_floor,mean(spread_losses))
    floor_w=isempty(w.values) ? NaN : max(numerical_floor,mean(w.values))
    summarize(v)=all(isfinite,v) ? (q05=quantile(v,.05),median=median(v),q95=quantile(v,.95),mean=mean(v),n=length(v)) :
        (q05=NaN,median=NaN,q95=NaN,mean=NaN,n=length(v))
    summaries=(spread=summarize(spreads),effective_dimension=summarize(dimensions),
        spread_loss=summarize(spread_losses),coverage50=summarize(cover50),coverage90=summarize(cover90),
        coverage90_absdev=summarize(covdev))
    (;spread_reference=sref,floor_w,floor_s,summaries,w1_status=w.status,
        w1_summary=isempty(w.values) ? nothing : summarize(w.values),
        draws=(spread=spreads,effective_dimension=dimensions,spread_loss=spread_losses,
            coverage50=cover50,coverage90=cover90,coverage90_absdev=covdev,w1=w.values),
        seed,scope="Conditional empirical reference draws; separate target/check roles; mean reference losses are normalization scales, not universal error floors or confidence intervals")
end
end
