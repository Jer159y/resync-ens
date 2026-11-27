using Test, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__,"..","kernels","resync_v4_metrics.jl"))
using .ResyncV4Metrics
@testset "v4 finite empirical metrics and protected selection" begin
    rng=MersenneTwister(61)
    x=[-2.,0.,1.,4.];y=.3
    direct=mean(abs.(x.-y))-sum(abs(a-b) for a in x,b in x)/(2length(x)^2)
    @test empirical_crps(x,y)≈direct
    @test empirical_crps(fill(2.,4),0.)==2.
    @test isnan(empirical_crps([1.,NaN],0.))
    @test wasserstein1([0.,2.],[0.,1.,2.,3.])≈.5
    @test wasserstein1([0.,0.],[0.])==0.
    @test wasserstein1([0.,NaN],[0.])==Inf
    A=randn(rng,2,12,6)
    @test window_spread(A)≈sqrt(mean(var(A;dims=3,corrected=true)))
    amplitudes=reshape([.1,10.],1,2,1)
    oscillating=repeat(reshape([-1.,1.],1,1,2),1,2,1).*amplitudes
    @test !isapprox(abs(log(window_spread(oscillating))),mean(abs.(log.([window_spread(oscillating[:,t:t,:]) for t in 1:2]))))
    Q=Matrix(qr(hcat(ones(8),randn(rng,8,6))).Q)[:,2:7]
    independent=reshape(Q,1,8,6)
    @test trajectory_dimension(independent).effective_dimension≈6.
    copied=copy(independent);copied[:,:,5]=copied[:,:,1];copied[:,:,6]=copied[:,:,2]
    @test trajectory_dimension(copied).effective_dimension≈36/10
    @test trajectory_dimension(cat(A[:,:,1:1],-A[:,:,1:1];dims=3)).effective_dimension≈1.
    @test trajectory_dimension(A).effective_dimension>2 # D alone is not a trajectory rank bound.
    constant=copy(A);constant[:,:,2].=0
    d=trajectory_dimension(constant)
    @test isnan(d.effective_dimension) && d.n_eligible==5 && d.excluded_constant==1
    @test isfinite(d.eligible_dimension)
    bad=copy(A);bad[1,2,1]=NaN
    @test window_spread(bad)==Inf
    @test trajectory_dimension(bad).excluded_nonfinite==1
    truth=zeros(2,12);truth[1,1]=NaN
    p=probability_metrics(A,truth)
    @test p.n_scored==23 && p.excluded_missing_truth==1
    @test p.coverage90_absdev≈mean(abs.(p.coverage90_by_coordinate.-.9))
    @test !probability_metrics(bad,truth).output_valid
    @test isnan(probability_metrics(bad,truth).crps)
    @test_throws DimensionMismatch probability_metrics(A,zeros(1,12))
    @test_throws ErrorException normalize_training(ones(2,10))
    ref=reference_starts(MersenneTwister(1),500,10,30)
    @test length(ref)==30 && minimum(diff(sort(ref)))>=10
    @test reference_starts(rng,299,10,30)===nothing
    row(k,m,w,s;valid=true)=(k=k,multiplier=m,w1=w,spread_loss=s,valid=valid)
    base=row(0,1,1.,1.);win=row(6,2,.8,.7);fail=row(1,1,.1,.1;valid=false)
    @test select_scheme([base,win,fail]).selected==win
    @test select_scheme([base,row(1,1,1.,1.)]).selected==base
    @test select_scheme([base,row(1,1,.5,1.2)]).selected==base
    @test select_scheme([base,row(6,1,.8,.8),row(3,1,.8,.8)]).selected.k==3
    @test select_scheme([base];floors=(2.,2.)).baseline_score==.5
    @test_throws ErrorException select_scheme([win])
    @test_throws ErrorException select_scheme([row(0,1,1.,1.;valid=false),win])
    @test_throws ErrorException select_scheme([base,base])
    # Large finite outcomes are not confused with a historical failure penalty.
    @test length(scheme_front([row(0,1,1e7,1e7)]))==1
end
