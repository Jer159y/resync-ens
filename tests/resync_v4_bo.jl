using Test
include(joinpath(@__DIR__,"../kernels/resync_v4_bo.jl"))
using .ResyncV4BO
domain=[(k=k,multiplier=m) for k in [1,3,6,15,30,60,90] for m in [1,2,4]]
initial=[(k=30,multiplier=1),(k=6,multiplier=2),(k=60,multiplier=4),(k=90,multiplier=1)]
row(c,valid=true,w=.5,s=.2)=(;c...,valid=valid,w1=w,spread_loss=s)
@testset "v4 BO failures and reproducibility" begin
    h=[row((k=0,multiplier=1))]
    @test next_candidate(h,domain,initial;scales=(.5,.2)).candidate==first(initial)
    append!(h,[row(c,false,Inf,NaN) for c in initial])
    n=next_candidate(h,domain,initial;scales=(.5,.2))
    @test isfinite(n.acquisition) && 0<n.feasibility<1
    @test n.method=="distance_exploration_times_feasibility"
    @test n==next_candidate(h,domain,initial;scales=(.5,.2))
    push!(h,row(n.candidate,true,.3,.15))
    gp=next_candidate(h,domain,initial;scales=(.5,.2))
    @test isfinite(gp.acquisition) && 0<gp.feasibility<1
    @test gp==next_candidate(h,domain,initial;scales=(.5,.2))
    @test !any(r->r.k==gp.candidate.k&&r.multiplier==gp.candidate.multiplier,h)
    bad=copy(h);bad[1]=row((k=0,multiplier=1),false,Inf,Inf)
    @test_throws ErrorException next_candidate(bad,domain,initial;scales=(.5,.2))
    @test_throws ErrorException next_candidate(h,domain,initial;scales=(0.,.2))
    for _ in 1:6
        n=next_candidate(h,domain,initial;scales=(.5,.2));push!(h,row(n.candidate,true,.4,.1))
    end
    @test length(unique((r.k,r.multiplier) for r in h))==length(h)
end
