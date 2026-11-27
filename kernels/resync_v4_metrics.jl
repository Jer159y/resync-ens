module ResyncV4Metrics
using Statistics, LinearAlgebra, Random
export empirical_crps, wasserstein1, window_spread, trajectory_dimension,
       probability_metrics, marginal_w1, scheme_front, select_scheme,
       reference_starts, normalize_training, window_metrics

"Fixed training-only population scales; constant coordinates require an explicit protocol decision."
function normalize_training(train::AbstractMatrix)
    all(isfinite, train) || error("Nonfinite training coordinates")
    mu = vec(mean(train; dims=2))
    sd = vec(std(train; dims=2, corrected=false))
    all(x -> isfinite(x) && x > 0, sd) || error("Degenerate training scale")
    (; mu, sd)
end

"CRPS of the issued finite empirical distribution, with denominator N^2."
function empirical_crps(x::AbstractVector, y::Real)
    n = length(x)
    n > 0 || throw(ArgumentError("Empty ensemble"))
    all(isfinite, x) && isfinite(y) || return NaN
    s = sort(Float64.(x))
    mean(abs.(s .- y)) - sum((2i-n-1)*s[i] for i in 1:n)/n^2
end

"Exact W1 for two unweighted empirical measures of possibly unequal size."
function wasserstein1(x::AbstractVector, y::AbstractVector)
    isempty(x) || isempty(y) ? throw(ArgumentError("Empty distribution")) : nothing
    all(isfinite, x) && all(isfinite, y) || return Inf
    a=sort(Float64.(x)); b=sort(Float64.(y)); n=length(a); m=length(b)
    i=j=1; left=total=0.0
    while i<=n && j<=m
        ca=i/n; cb=j/m; right=min(ca,cb)
        total += (right-left)*abs(a[i]-b[j])
        left=right
        ca<=right && (i+=1)
        cb<=right && (j+=1)
    end
    total
end

"Native-time RMS of unbiased member variance, A is D x time x members, already standardized."
function window_spread(A::AbstractArray{<:Real,3})
    D,T,N=size(A)
    min(D,T)>0 && N>=2 || throw(ArgumentError("Invalid window dimensions"))
    all(isfinite,A) || return Inf
    sqrt(mean(var(A; dims=3, corrected=true)))
end

"Trajectory correlation dimension; never substitute a filtered subset for the full population."
function trajectory_dimension(A::AbstractArray{<:Real,3}; samples::Int=64, constant_tol::Real=1e-10)
    D,T,N=size(A)
    D>=1 && T>=2 && N>=2 && samples>=2 && isfinite(constant_tol) && constant_tol>=0 ||
        throw(ArgumentError("Invalid dimension/tolerance"))
    idx=unique(round.(Int,range(1,T;length=min(T,samples))))
    X=Float64.(A[:,idx,:]); q=length(idx)
    nonfinite=[!all(isfinite,view(A,:,:,i)) for i in 1:N]
    constant=falses(N); vectors=Vector{Vector{Float64}}()
    for i in 1:N
        nonfinite[i] && continue
        B=Matrix(view(X,:,:,i)); B .-= mean(B;dims=2)
        amplitude=sqrt(mean(abs2,B))
        if amplitude<=constant_tol
            constant[i]=true
        else
            push!(vectors,vec(B)./norm(B))
        end
    end
    eligible=length(vectors)
    # The correlation matrix of a single nonconstant member is well defined.
    subset=if eligible==0
        NaN
    else
        V=hcat(vectors...); G=V'*V
        eligible^2/sum(abs2,G)
    end
    full=eligible==N ? subset : NaN
    (; effective_dimension=full, eligible_dimension=subset, n_total=N,
       n_eligible=eligible, excluded_constant=count(constant),
       excluded_nonfinite=count(nonfinite), rank_bound=min(eligible,D*(q-1)), n_sample_times=q)
end

"Coordinate- and time-matched scores; missing truth is excluded and counted, never interpolated."
function probability_metrics(A::AbstractArray{<:Real,3}, truth::AbstractMatrix; stride::Int=1)
    D,T,N=size(A)
    size(truth)==(D,T) && N>=2 && stride>=1 || throw(DimensionMismatch("Prediction/truth alignment"))
    finite=all(isfinite,A)
    c50=zeros(D); c90=zeros(D); count_d=zeros(Int,D)
    crps=0.; n=0; missing_truth=0
    for t in 1:stride:T, d in 1:D
        y=truth[d,t]
        if !isfinite(y)
            missing_truth+=1
            continue
        end
        finite || continue
        x=collect(view(A,d,t,:))
        crps+=empirical_crps(x,y); n+=1; count_d[d]+=1
        # Statistics.quantile uses the explicitly chosen Hyndman-Fan type 7 convention.
        lo50,hi50=quantile(x,[.25,.75];alpha=1.,beta=1.)
        lo90,hi90=quantile(x,[.05,.95];alpha=1.,beta=1.)
        c50[d]+=lo50<=y<=hi50; c90[d]+=lo90<=y<=hi90
    end
    cov50=[count_d[d]>0 ? c50[d]/count_d[d] : NaN for d in 1:D]
    cov90=[count_d[d]>0 ? c90[d]/count_d[d] : NaN for d in 1:D]
    # An unobserved coordinate keeps the coordinate-mean score undefined.
    (; crps=n>0 ? crps/n : NaN, coverage50=mean(cov50), coverage90=mean(cov90),
       coverage90_absdev=mean(abs.(cov90.-.9)), coverage50_by_coordinate=cov50,
       coverage90_by_coordinate=cov90, n_scored=n, excluded_missing_truth=missing_truth,
       n_scored_by_coordinate=count_d, output_valid=finite)
end

function marginal_w1(A::AbstractArray{<:Real,3}, target::AbstractMatrix)
    size(A,1)==size(target,1) || throw(DimensionMismatch("Target coordinates"))
    mean(wasserstein1(vec(A[d,:,:]),vec(target[d,:])) for d in axes(A,1))
end

function window_metrics(A,truth; probability_stride=1, dimension_samples=64, constant_tol=1e-10)
    merge((spread=window_spread(A),),
        trajectory_dimension(A;samples=dimension_samples,constant_tol),
        probability_metrics(A,truth;stride=probability_stride))
end

numeric(r)=r.valid && all(isfinite,(r.w1,r.spread_loss)) && min(r.w1,r.spread_loss)>=0
function scheme_front(rows)
    valid=filter(numeric,rows)
    [r for r in valid if !any(s->s.w1<=r.w1 && s.spread_loss<=r.spread_loss &&
        (s.w1<r.w1 || s.spread_loss<r.spread_loss),valid)]
end

"Frozen baseline scales and minimax. Failure is an explicit flag, never a sentinel threshold."
function select_scheme(rows; floors=(1e-12,1e-12), atol=1e-10, rtol=1e-6)
    all(x->isfinite(x)&&x>0,floors) && atol>=0 && rtol>=0 || error("Invalid frozen floors/tolerance")
    bases=filter(r->r.k==0,rows)
    length(bases)==1 || error("Exactly one paired baseline required")
    base=only(bases); numeric(base) || error("Invalid paired baseline; selection blocked")
    valid=filter(numeric,rows)
    scales=(max(base.w1,floors[1]),max(base.spread_loss,floors[2]))
    score(r)=max(r.w1/scales[1],r.spread_loss/scales[2])
    best=minimum(score.(valid))
    tied=filter(r->abs(score(r)-best)<=atol+rtol*abs(best),valid)
    selected=first(sort(tied;by=r->(r.k==0 ? 0 : 1,r.k/r.multiplier,r.k,r.multiplier)))
    (; selected, scales, selected_score=score(selected),baseline_score=score(base),front=scheme_front(rows))
end

"Nonoverlapping, random-phase reference window selection. No replacement within a draw."
function reference_starts(rng::AbstractRNG,T::Int,L::Int,N::Int)
    L>=1 && N>=2 || throw(ArgumentError("Invalid reference dimensions"))
    T>=N*L || return nothing
    phase=rand(rng,0:min(L-1,T-N*L))
    pool=collect((phase+1):L:(T-L+1))
    pool[randperm(rng,length(pool))[1:N]]
end
end
