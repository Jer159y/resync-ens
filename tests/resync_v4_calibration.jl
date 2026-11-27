using Test, Random, Statistics
include(joinpath(@__DIR__,"..","kernels","resync_v4_calibration.jl"))
using .ResyncV4Calibration
using .ResyncV4Metrics
@testset "v4 matched calibration and exact pooled references" begin
    rng=MersenneTwister(9);target=randn(rng,2,80);pool=randn(rng,2,400);checks=randn(rng,2,40)
    result=calibrate(target,pool,checks,4,5;fit_draws=8,check_draws=8,distribution_draws=4)
    @test result.w1_status=="computed"
    @test result.floor_s≈mean(result.draws.spread_loss)
    @test result.floor_w≈mean(result.draws.w1)
    @test result.spread_reference≈median(result.draws.spread)
    @test length(result.draws.effective_dimension)==8
    @test 0<=result.summaries.coverage90.mean<=1
    # Recreate one selected ensemble independently to check the sorted-pool optimization.
    direct_rng=MersenneTwister(73)
    starts=reference_starts(direct_rng,size(pool,2),40,5)
    A=ResyncV4Calibration.ensemble_window(pool,starts,40)
    fast=distribution_reference(target,pool,5,40;draws=1,seed=73)
    @test only(fast.values)≈marginal_w1(A,target)
    @test distribution_reference(target,pool[:,1:199],5,40;draws=2).status=="unavailable"
    bad=copy(pool);bad[1,1]=NaN
    @test_throws ErrorException calibrate(target,bad,checks,4,5;fit_draws=2,check_draws=2,distribution_draws=2)
end
