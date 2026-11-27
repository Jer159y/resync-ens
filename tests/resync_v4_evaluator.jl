using Test, Statistics
include(joinpath(@__DIR__,"..","kernels","resync_v4_evaluator.jl"))
using .ResyncV4Evaluator
@testset "v4 fixed-window streaming integration" begin
    A=repeat(reshape([-1.,0.,1.],1,1,3),1,4,1)
    truth=zeros(1,4)
    e=WindowEvaluator(4,3,40,reshape([-1.,0.,1.],1,3),1.;probability_stride=1)
    @test_throws ErrorException finish(e)
    for _ in 1:40;push_base!(e,A,truth);end
    s=finish(e)
    @test s.valid && s.w1≈0. && s.spread_loss≈0.
    @test s.crps≈2/9
    @test s.coverage90==1. && s.coverage90_absdev≈.1
    @test s.n_base_windows==40 && s.n_distribution_windows==4
    @test isnan(s.effective_dimension) # Undefined correlation is not a hidden selection gate.
    @test s.mean_eligible_members==0.
    @test_throws ErrorException push_base!(e,A,truth)
    @test_throws DimensionMismatch push_base!(WindowEvaluator(4,3,10,reshape([-1.,0.,1.],1,3),1.),A[:,1:3,:],truth)
    f=WindowEvaluator(4,3,10,reshape([-1.,0.,1.],1,3),1.)
    bad=copy(A);bad[1,2,1]=Inf
    push_base!(f,bad,truth)
    for _ in 2:10;push_base!(f,A,truth);end
    @test !finish(f).valid && finish(f).nonfinite_returned_values==1
    z=WindowEvaluator(4,3,10,reshape([-1.,0.,1.],1,3),1.)
    for _ in 1:10;push_base!(z,zeros(1,4,3),truth);end
    @test !finish(z).valid # Finite but zero spread has infinite declared selection loss.
    @test_throws ErrorException WindowEvaluator(4,3,11,reshape([-1.,0.,1.],1,3),1.)
end
