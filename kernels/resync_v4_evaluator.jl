# Streaming fixed-base-window observer. Inputs must already use the frozen training transform.
if !isdefined(@__MODULE__,:ResyncV4Metrics)
    include(joinpath(@__DIR__,"resync_v4_metrics.jl"))
end
module ResyncV4Evaluator
using Statistics
using ..ResyncV4Metrics
export WindowEvaluator, push_base!, finish

mutable struct WindowEvaluator
    base::Int
    members::Int
    expected_bases::Int
    spread_reference::Float64
    target::Matrix{Float64}
    probability_stride::Int
    dimension_samples::Int
    constant_tol::Float64
    pending::Array{Float64,3}
    filled::Int
    n_bases::Int
    returned_finite::Bool
    nonfinite_values::Int
    base_rows::Vector{NamedTuple}
    distribution_rows::Vector{NamedTuple}
end

function WindowEvaluator(base::Int,members::Int,expected_bases::Int,target::AbstractMatrix,
        spread_reference::Real;probability_stride=max(1,base÷20),dimension_samples=64,constant_tol=1e-10)
    base>=2 && members>=2 && expected_bases>0 && expected_bases%10==0 || error("Invalid fixed scoring grid")
    all(isfinite,target) && size(target,2)>0 || error("Invalid calibration target")
    isfinite(spread_reference) && spread_reference>0 || error("Invalid calibrated spread reference")
    probability_stride>=1 && dimension_samples>=2 && constant_tol>=0 || error("Invalid frozen numerics")
    WindowEvaluator(base,members,expected_bases,Float64(spread_reference),Matrix{Float64}(target),
        probability_stride,dimension_samples,Float64(constant_tol),
        Array{Float64,3}(undef,size(target,1),10base,members),0,0,true,0,NamedTuple[],NamedTuple[])
end

"Consume exactly one base window; no raw outputs are written by this observer."
function push_base!(e::WindowEvaluator,A::AbstractArray{<:Real,3},truth::AbstractMatrix)
    size(A)==(size(e.target,1),e.base,e.members) || throw(DimensionMismatch("Complete base window required"))
    size(truth)==(size(e.target,1),e.base) || throw(DimensionMismatch("Truth window misaligned"))
    e.n_bases<e.expected_bases || error("Too many base windows")
    metrics=window_metrics(A,truth;probability_stride=e.probability_stride,
        dimension_samples=e.dimension_samples,constant_tol=e.constant_tol)
    e.n_bases+=1
    e.returned_finite &= metrics.output_valid
    e.nonfinite_values+=count(x->!isfinite(x),A)
    loss=isfinite(metrics.spread) && metrics.spread>0 ? abs(log(metrics.spread/e.spread_reference)) : Inf
    push!(e.base_rows,merge((base_index=e.n_bases,start_step=(e.n_bases-1)*e.base+1,
        stop_step=e.n_bases*e.base,spread_loss=loss),metrics))
    e.pending[:,(e.filled+1):(e.filled+e.base),:].=A
    e.filled+=e.base
    if e.filled==10 * e.base
        w=marginal_w1(e.pending,e.target)
        push!(e.distribution_rows,(start_base=e.n_bases-9,stop_base=e.n_bases,w1=w))
        e.filled=0
    end
    last(e.base_rows)
end

"Finalize only complete forecasts; retain undefined auxiliary values rather than filtering."
function finish(e::WindowEvaluator)
    e.n_bases==e.expected_bases && e.filled==0 || error("Incomplete forecast scoring windows")
    w=mean(r.w1 for r in e.distribution_rows)
    s=mean(r.spread_loss for r in e.base_rows)
    valid=e.returned_finite && all(isfinite,(w,s))
    (;valid,w1=w,spread_loss=s,spread=mean(r.spread for r in e.base_rows),
      crps=mean(r.crps for r in e.base_rows),
      coverage90=mean(r.coverage90 for r in e.base_rows),
      coverage50=mean(r.coverage50 for r in e.base_rows),
      coverage90_absdev=mean(r.coverage90_absdev for r in e.base_rows),
      effective_dimension=mean(r.effective_dimension for r in e.base_rows),
      eligible_dimension=mean(r.eligible_dimension for r in e.base_rows),
      mean_eligible_members=mean(r.n_eligible for r in e.base_rows),
      excluded_missing_truth=sum(r.excluded_missing_truth for r in e.base_rows),
      nonfinite_returned_values=e.nonfinite_values,n_base_windows=e.n_bases,
      n_distribution_windows=length(e.distribution_rows))
end
end
