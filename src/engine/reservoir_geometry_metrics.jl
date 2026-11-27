# ============================================================
# reservoir_geometry_metrics.jl
#
# Statement C용 새 blind selection 후보 지표: "이 reservoir의 test rollout이
# 자기 자신의 training-state geometry와 얼마나 잘 어울리는가"를 ground truth
# 없이 평가해서 reservoir(ensemble member)들을 ranking한다.
#
# 이미 존재하는 reservoir dynamics/학습/predict 코드는 전혀 새로 만들지
# 않는다 -- 이 파일은 순수하게 이미 만들어진 reservoir state matrix
#   X_train :: N × T_train   (node 수 × training-phase time step)
#   X_test  :: N × T_test    (같은 reservoir가 만든 test-phase trajectory)
# 를 입력으로 받아서 분석만 하는 함수들이다. `X_train`/`X_test`는 이 파일이
# 만들지 않는다 -- 호출하는 쪽이 기존 `ReservoirComputing.collectstates` 등을
# 이용해 준비해서 넘긴다.
#
# 독립적인 세 가지 metric (모두 "값이 작을수록 좋음"):
#   A. leverage_distance             : training SVD 기준 ridge-leverage
#                                       score 분포의 train-vs-test
#                                       Wasserstein-1 거리
#   B. subspace_residual              : training 상위-k principal subspace
#                                       (분산 누적 `variance_threshold`
#                                       기준)로 test state를 projection했을
#                                       때 남는 평균 residual 에너지
#   C. effective_dimension_mismatch   : ridge effective dimension
#                                       (sum s^2/(s^2+λ), train/test 각자
#                                       고유의 SVD)의 log-ratio 절댓값
#
# `evaluate_reservoir_geometry`가 reservoir 하나에 대한 세 metric을 모두
# 계산하고, `rank_reservoirs`/`rank_reservoirs_all_metrics`가 여러
# reservoir의 결과를 metric별로 독립적으로 ranking한다 (세 metric을 절대
# 합치지 않음). `combined_geometry_score`는 그것과는 별도의 실험으로 쓰라고
# 만든 optional weighted-sum ranking이다.
#
# `training_geometry_profile(X_train)` / `score_against_profile(profile, X_test)`는
# `evaluate_reservoir_geometry`를 training-side/test-side로 쪼갠 것이다 (StatementC.jl이
# 이 7번째 selector를 segment마다 반복 호출할 때, 매번 X_train의 SVD를 다시 계산하지
# 않고 멤버당 한 번만 계산한 profile을 재사용하기 위함). `evaluate_reservoir_geometry`는
# 이제 이 둘을 이어붙인 thin wrapper이며 반환값/수치는 이전과 완전히 동일하다.
# ============================================================

using LinearAlgebra
using Statistics

# ------------------------------------------------------------
# 1. Training-referenced normalization
# ------------------------------------------------------------

"""
    train_zscore(X_train; epsilon=1e-8) -> (Z_train, μ, σ)

`X_train`(N × T_train)의 node별(dims=2) mean/std로 `X_train` 자신을 정규화한다.
분산이 0에 가깝거나 정확히 0인(dead/saturated) node에서 나눗셈이 터지지 않도록
표준편차 분모에 `epsilon`을 더한다.
"""
function train_zscore(X_train::AbstractMatrix; epsilon::Float64=1e-8)
    Xtr = Float64.(X_train)
    μ = vec(mean(Xtr; dims=2))
    σ = vec(std(Xtr; dims=2, corrected=false))
    Z_train = (Xtr .- μ) ./ (σ .+ epsilon)
    return Z_train, μ, σ
end

"""
    train_test_zscore(X_train, X_test; epsilon=1e-8) -> (Z_train, Z_test, μ, σ)

`X_train`(N × T_train)에서만 node별(dims=2) mean/std를 계산해 `X_train`과
`X_test`를 동일한 통계로 정규화한다 -- `X_test` 자신의 통계는 쓰지 않는다.
"""
function train_test_zscore(
    X_train::AbstractMatrix,
    X_test::AbstractMatrix;
    epsilon::Float64=1e-8,
)
    size(X_train, 1) == size(X_test, 1) || throw(DimensionMismatch(
        "X_train and X_test must have the same number of reservoir nodes (rows): " *
        "got $(size(X_train,1)) vs $(size(X_test,1))"
    ))

    Z_train, μ, σ = train_zscore(X_train; epsilon=epsilon)
    Xte = Float64.(X_test)
    Z_test = (Xte .- μ) ./ (σ .+ epsilon)

    return Z_train, Z_test, μ, σ
end

# ------------------------------------------------------------
# 2. Training geometry SVD
# ------------------------------------------------------------

"""
    training_geometry_svd(Z_train) -> SVD

`Z_train = U * Diagonal(S) * V'`의 thin SVD(`full=false`). `U`의 column이
training reservoir-state space의 principal direction이며, 이 orientation을
아래 모든 metric(leverage / subspace projection / effective dimension)에서
그대로 재사용한다.
"""
training_geometry_svd(Z_train::AbstractMatrix) = svd(Z_train; full=false)

# ------------------------------------------------------------
# 3. Metric A -- ridge leverage distribution distance
# ------------------------------------------------------------

"""
    ridge_leverage_scores(Z, U, S; λ) -> Vector{Float64}

Training SVD의 `U`/`S` (`Z_train = U Σ V'`) 기준으로 `Z`의 각 column `z_t`에 대해
ridge-leverage-like score `sum_j (u_j' z_t)^2 / (s_j^2 + λ)`를 계산한다.
`(Z_train' Z_train + λI)^{-1}`를 explicit하게 만들지 않고 SVD만으로 계산한다. `V`/`Vt`는
이 계산에 필요 없으므로 인자로 받지 않는다 -- `training_geometry_profile`이 `Vt`를 캐싱하지
않는 이유이기도 하다 (member당 캐시 크기를 지배하는 건 `Vt`인데, quantile로 이미 줄여 둔
`train_leverage_quantiles`와 달리 아예 안 쓰는 값이었다).
"""
function ridge_leverage_scores(Z::AbstractMatrix, U::AbstractMatrix, S::AbstractVector; λ::Float64)
    λ > 0 || throw(ArgumentError("λ must be positive"))
    proj = U' * Z                # (rank × T): row j = u_j' z_t
    denom = S .^ 2 .+ λ
    return vec(sum((proj .^ 2) ./ denom; dims=1))
end

# Quantile-matched 1D Wasserstein-1 distance is now DynamicsMetrics.jl's
# `QuantileWassersteinDistance` (ported from this file's original
# quantile_wasserstein_distance); score_against_profile below inlines the
# same quantile-comparison math directly since it reuses a precomputed
# quantile vector (training_geometry_profile.train_leverage_quantiles)
# rather than quantizing both sides fresh on every call.

# ------------------------------------------------------------
# 4. Metric B -- training subspace residual
# ------------------------------------------------------------

"""
    training_subspace_dimension(F; threshold=0.99) -> Int

Training SVD singular value의 cumulative explained variance가 `threshold`
이상이 되는 최소 차원 k.
"""
function training_subspace_dimension(F::SVD; threshold::Float64=0.99)
    0 < threshold <= 1 || throw(ArgumentError("threshold must be in (0, 1]"))
    s2 = F.S .^ 2
    total = sum(s2)
    total > 0 || return length(F.S)
    cum = cumsum(s2) ./ total
    k = findfirst(>=(threshold), cum)
    return isnothing(k) ? length(F.S) : k
end

"""
    subspace_projection_residuals(Z, Uk) -> Vector{Float64}

`Uk`(N × k, training subspace의 orthonormal basis)로 `Z`의 각 column을
projection한 뒤 남는 residual 에너지 `‖z_t - U_k U_k' z_t‖²`.
"""
function subspace_projection_residuals(Z::AbstractMatrix, Uk::AbstractMatrix)
    proj = Uk' * Z
    recon = Uk * proj
    return vec(sum(abs2, Z .- recon; dims=1))
end

# ------------------------------------------------------------
# 5. Metric C -- effective dimension mismatch
# ------------------------------------------------------------

"""
    ridge_effective_dimension(S; λ) -> Float64

Singular value 벡터 `S`로부터 ridge effective dimension
`sum_j s_j^2 / (s_j^2 + λ)`.
"""
function ridge_effective_dimension(S::AbstractVector; λ::Float64)
    λ > 0 || throw(ArgumentError("λ must be positive"))
    s2 = S .^ 2
    return sum(s2 ./ (s2 .+ λ))
end

# ------------------------------------------------------------
# Per-reservoir evaluation: bundles metrics A/B/C
# ------------------------------------------------------------

"""
    training_geometry_profile(X_train; λ=1e-6, variance_threshold=0.99,
                               epsilon=1e-8, n_quantiles=200) -> NamedTuple

`X_train`(N × T_train) 하나로부터, 이 reservoir의 training-state geometry를
`score_against_profile`로 반복 재사용할 수 있도록 필요한 모든 것을 미리 계산해
둔다: z-score 통계(`μ`, `σ`), training SVD의 `U`/`S`(`U_train`, `S_train` -- `Vt`는 아무 데도
쓰이지 않으므로 저장하지 않는다), Metric B의 subspace
차원/basis(`k`, `Uk`), Metric C의 raw ridge effective dimension(`d_eff_train`),
그리고 (원본 `train_leverage` 벡터 전체 대신) `quantile_wasserstein_distance`가
실제로 필요로 하는 quantile들만 미리 뽑아 둔 `train_leverage_quantiles` -- 멤버
수천~만 단위로 캐싱할 때 저장 용량을 줄인다. `λ`/`epsilon`/`n_quantiles`도
profile에 함께 저장해서, 나중에 `score_against_profile`이 training 쪽과
다른 값을 실수로 쓰는 일이 없게 한다.
"""
function training_geometry_profile(
    X_train::AbstractMatrix;
    λ::Float64=1e-6,
    variance_threshold::Float64=0.99,
    epsilon::Float64=1e-8,
    n_quantiles::Int=200,
)
    Z_train, μ, σ = train_zscore(X_train; epsilon=epsilon)
    F_train = training_geometry_svd(Z_train)

    train_leverage = ridge_leverage_scores(Z_train, F_train.U, F_train.S; λ=λ)
    probs = range(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1); length=n_quantiles)
    train_leverage_quantiles = quantile(vec(train_leverage), probs)

    k = training_subspace_dimension(F_train; threshold=variance_threshold)
    Uk = Matrix(F_train.U[:, 1:k])

    d_eff_train = ridge_effective_dimension(F_train.S; λ=λ)

    # `F_train` itself (in particular `F_train.Vt`, N × T_train) is deliberately not stored:
    # nothing downstream ever reads `V`/`Vt` (see `ridge_leverage_scores` above), so caching the
    # full `SVD` struct would balloon this profile by a `T_train`-sized matrix per member for no
    # reason -- at ensemble scale (thousands of members) that's the difference between a profile
    # cache of a few hundred MB and one of tens/hundreds of GB. Only the thin `U`/`S` that are
    # actually used are kept, as plain arrays (also avoids caching a `LinearAlgebra.SVD` struct
    # through JLD2's generic struct serialization).
    return (
        μ=μ, σ=σ, U_train=Matrix(F_train.U), S_train=F_train.S, k=k, Uk=Uk,
        d_eff_train=d_eff_train,
        train_leverage_quantiles=train_leverage_quantiles,
        λ=λ, epsilon=epsilon, n_quantiles=n_quantiles,
    )
end

"""
    profile_zscore(profile, X_test) -> Z_test

`training_geometry_profile`이 저장해 둔 training 통계(`μ`/`σ`)만으로 `X_test`를
정규화한다 (`X_test` 자신의 통계는 쓰지 않음). 세 metric 함수(`leverage_distance_score`/
`subspace_residual_score`/`effective_dimension_mismatch_score`) 모두 이 `Z_test`를
입력으로 받는다 -- 정규화를 metric 개수만큼 반복하지 않고 한 번만 하기 위함.
"""
function profile_zscore(profile::NamedTuple, X_test::AbstractMatrix)
    size(X_test, 1) == length(profile.μ) || throw(DimensionMismatch(
        "X_test must have the same number of reservoir nodes (rows) as the training " *
        "profile: got $(size(X_test,1)) vs $(length(profile.μ))"
    ))
    Xte = Float64.(X_test)
    return (Xte .- profile.μ) ./ (profile.σ .+ profile.epsilon)
end

"""
    leverage_distance_score(profile, Z_test) -> Float64

Metric A만 계산: `Z_test`(이미 `profile_zscore`로 정규화된 상태)의 ridge-leverage
score 분포와 training 분포(quantile-비교, `profile.train_leverage_quantiles`) 사이의
quantile-Wasserstein 거리. Metric B/C에 필요한 계산(subspace projection, SVD)은
전혀 하지 않는다 -- `reservoir_geometry_leverage` selector처럼 이 metric만 필요한
호출자를 위한 것.
"""
function leverage_distance_score(profile::NamedTuple, Z_test::AbstractMatrix)
    # both sides already quantile-reduced to the same probability grid, so this is
    # exactly quantile_wasserstein_distance(train_leverage, test_leverage; n_quantiles)
    # without re-quantizing train_leverage every call.
    test_leverage = ridge_leverage_scores(Z_test, profile.U_train, profile.S_train; λ=profile.λ)
    probs = range(
        1 / (profile.n_quantiles + 1), profile.n_quantiles / (profile.n_quantiles + 1);
        length=profile.n_quantiles,
    )
    test_leverage_quantiles = quantile(vec(test_leverage), probs)
    return mean(abs.(profile.train_leverage_quantiles .- test_leverage_quantiles))
end

"""
    subspace_residual_score(profile, Z_test) -> Float64

Metric B만 계산: training 상위-k subspace(`profile.Uk`)로 `Z_test`를 projection했을 때
남는 평균 residual 에너지. Metric A/C에 필요한 계산(leverage, SVD)은 전혀 하지 않는다.
"""
function subspace_residual_score(profile::NamedTuple, Z_test::AbstractMatrix)
    return mean(subspace_projection_residuals(Z_test, profile.Uk))
end

"""
    effective_dimension_mismatch_score(profile, Z_test) -> (mismatch, d_eff_test)

Metric C만 계산: `Z_test`(X_test's own SVD, not projected onto the training basis) 기준
raw ridge effective dimension `d_eff_test`와, training 값(`profile.d_eff_train`) 대비
log-ratio 절댓값 `mismatch`. 이 함수가 세 metric 중 유일하게 SVD를 계산한다 -- 계산량
대부분이 여기 있으므로, leverage/subspace만 필요한 호출자는 이 함수를 아예 부르지
않아야 한다.
"""
function effective_dimension_mismatch_score(profile::NamedTuple, Z_test::AbstractMatrix)
    F_test = svd(Z_test; full=false)
    d_eff_test = ridge_effective_dimension(F_test.S; λ=profile.λ)
    mismatch = abs(log((d_eff_test + profile.epsilon) / (profile.d_eff_train + profile.epsilon)))
    return mismatch, d_eff_test
end

"""
    score_against_profile(profile, X_test) -> NamedTuple

`training_geometry_profile`이 미리 계산해 둔 training-side 통계만으로(training SVD를
다시 계산하지 않고), 새 `X_test`에 대한 세 독립 geometry metric(모두 작을수록 좋음)을
계산한다. `X_test`의 정규화는 `profile.μ`/`profile.σ`(training 통계)를 그대로 쓴다 --
`X_test` 자신의 통계는 쓰지 않는다. 세 metric 중 하나만 필요하면(예: blind selector가
metric 하나만 쓰는 경우) 이 함수 대신 `leverage_distance_score`/`subspace_residual_score`/
`effective_dimension_mismatch_score`를 `profile_zscore`와 함께 직접 쓰는 편이 낫다 --
이 함수는 세 metric을 모두 원하는 호출자(`evaluate_reservoir_geometry` 등)를 위한 것이다.

반환되는 NamedTuple: `leverage_distance`(Metric A), `subspace_residual`(Metric B,
trajectory 평균), `effective_dimension_mismatch`(Metric C), `d_eff_test`(Metric C의
원천이 되는, `X_test` 자신의 SVD 기준 raw ridge effective dimension).
"""
function score_against_profile(profile::NamedTuple, X_test::AbstractMatrix)
    Z_test = profile_zscore(profile, X_test)

    leverage_distance = leverage_distance_score(profile, Z_test)
    subspace_residual = subspace_residual_score(profile, Z_test)
    effective_dimension_mismatch, d_eff_test = effective_dimension_mismatch_score(profile, Z_test)

    return (
        leverage_distance=leverage_distance,
        subspace_residual=subspace_residual,
        effective_dimension_mismatch=effective_dimension_mismatch,
        d_eff_test=d_eff_test,
    )
end

"""
    evaluate_reservoir_geometry(X_train, X_test; λ=1e-6, variance_threshold=0.99,
                                 epsilon=1e-8, n_quantiles=200) -> NamedTuple

하나의 reservoir에 대해, test rollout이 그 reservoir 자신의 training-state
geometry와 얼마나 유사한지 세 개의 독립 metric(모두 작을수록 좋음)으로
평가한다. `training_geometry_profile` + `score_against_profile`을 이어붙인
thin wrapper -- 반환값/수치는 그 둘을 나누기 전과 완전히 동일하다. 반환되는
NamedTuple:

  - `leverage_distance`             : Metric A
  - `subspace_residual`             : Metric B (trajectory 평균)
  - `effective_dimension_mismatch`  : Metric C
  - `d_eff_train`, `d_eff_test`     : Metric C의 원천이 되는 raw ridge
                                       effective dimension (각자 고유의
                                       SVD 기준)
  - `subspace_dimension`            : Metric B에서 쓴 training subspace 차원 k
"""
function evaluate_reservoir_geometry(
    X_train::AbstractMatrix,
    X_test::AbstractMatrix;
    λ::Float64=1e-6,
    variance_threshold::Float64=0.99,
    epsilon::Float64=1e-8,
    n_quantiles::Int=200,
)
    profile = training_geometry_profile(
        X_train; λ=λ, variance_threshold=variance_threshold,
        epsilon=epsilon, n_quantiles=n_quantiles,
    )
    scores = score_against_profile(profile, X_test)
    return merge(scores, (d_eff_train=profile.d_eff_train, subspace_dimension=profile.k))
end

# ------------------------------------------------------------
# 6. Independent per-metric ranking (never combined in this section)
# ------------------------------------------------------------

"""
    rank_reservoirs(results, metric; names=eachindex(results)) -> Vector{<:NamedTuple}

`results`(각 원소가 `evaluate_reservoir_geometry`의 NamedTuple)를 단일
`metric`(`:leverage_distance` / `:subspace_residual` /
`:effective_dimension_mismatch` 중 하나) 기준으로만 ascending order(값이
작을수록 1등)로 정렬한다. 다른 metric은 전혀 참조하지 않는다 -- 세 metric을
합치는 것은 `combined_geometry_score`에서만 한다.
"""
function rank_reservoirs(
    results::AbstractVector{<:NamedTuple},
    metric::Symbol;
    names::AbstractVector=collect(eachindex(results)),
)
    length(results) == length(names) ||
        throw(DimensionMismatch("results and names must have the same length"))

    values = [Float64(getfield(r, metric)) for r in results]
    order = sortperm(values)
    return [(name=names[i], value=values[i], rank=rnk) for (rnk, i) in enumerate(order)]
end

"""
    rank_reservoirs_all_metrics(results; names=eachindex(results)) -> NamedTuple

세 독립 ranking(`rank_reservoirs`)을 한 번에 만든다. `by_leverage_distance`
/ `by_subspace_residual` / `by_effective_dimension_mismatch` 세 필드는 서로
섞이지 않는 별개의 ranking이다.
"""
function rank_reservoirs_all_metrics(
    results::AbstractVector{<:NamedTuple};
    names::AbstractVector=collect(eachindex(results)),
)
    return (
        by_leverage_distance=rank_reservoirs(results, :leverage_distance; names=names),
        by_subspace_residual=rank_reservoirs(results, :subspace_residual; names=names),
        by_effective_dimension_mismatch=rank_reservoirs(results, :effective_dimension_mismatch; names=names),
    )
end

# ------------------------------------------------------------
# 7. Optional combined score -- a separate experiment from section 6's
#    independent rankings above.
# ------------------------------------------------------------

"""
    robust_zscore(x; epsilon=1e-8) -> Vector{Float64}

Median/MAD 기반 robust normalization: `(x - median(x)) / (MAD(x) + epsilon)`.
"""
function robust_zscore(x::AbstractVector; epsilon::Float64=1e-8)
    m = median(x)
    mad = median(abs.(x .- m))
    return (x .- m) ./ (mad + epsilon)
end

"""
    combined_geometry_score(results; names=eachindex(results),
                             weights=(1.0, 1.0, 1.0), epsilon=1e-8) -> Vector{<:NamedTuple}

세 metric을 reservoir population 전체 기준 median/MAD로 각각
robust-normalize한 뒤 `weights`로 가중합한 단일 score를 만들어 ascending
order로 ranking한다 (작을수록 좋음). raw metric은 서로 스케일이 다르므로
직접 더하지 않는다. `weights`는 자유롭게 바꿀 수 있으며 기본값은 균등
`(1.0, 1.0, 1.0)`이다.

이 함수는 `rank_reservoirs_all_metrics`의 독립 ranking과는 별도의 실험으로
쓰라고 만들어졌다 -- 호출하지 않으면 combined score는 전혀 계산되지 않는다.
"""
function combined_geometry_score(
    results::AbstractVector{<:NamedTuple};
    names::AbstractVector=collect(eachindex(results)),
    weights::NTuple{3,Float64}=(1.0, 1.0, 1.0),
    epsilon::Float64=1e-8,
)
    length(results) == length(names) ||
        throw(DimensionMismatch("results and names must have the same length"))

    lev = robust_zscore(Float64[r.leverage_distance for r in results]; epsilon=epsilon)
    res = robust_zscore(Float64[r.subspace_residual for r in results]; epsilon=epsilon)
    dim = robust_zscore(Float64[r.effective_dimension_mismatch for r in results]; epsilon=epsilon)

    w1, w2, w3 = weights
    scores = w1 .* lev .+ w2 .* res .+ w3 .* dim

    order = sortperm(scores)
    return [(name=names[i], score=scores[i], rank=rnk) for (rnk, i) in enumerate(order)]
end
