# ============================================================
# resync_engine.jl
#
# StatementI_resync.jl(→ archive/statement_verification/statement_I_engine.jl)의 "full-history resync" 방식을
# 시스템 무관하게 옮긴 공용 엔진. R1~R5가 include한다 (이 파일은 top-level에서 rollout을 돌리지 않는다).
#
# 고정된 방식 (기본 옵션 = StatementI와 동일; tests/selftest.jl이 StatementI 엔진과 bit 단위로 대조):
#   segment s에서
#   1. 활성 멤버 전원이 carried state에서 τ step closed-loop rollout
#   2. bad criterion 점수(클수록 bad) 상위 K = round(p·n_active)개 + 비유한 멤버 = bad
#   3. bad 멤버마다 good pool에서 donor 배정 (:uniform = good[mod(j, |good|) + 1], :random)
#   4. bad 멤버 j는 "학습 직후 state"에서 donor의 누적 생성 이력을 teacher-forcing
#      (입력열 = [x0, y1, …, y_{T−1}], 다음 자율 입력 = y_T)
#   5. good 멤버는 자기 state 유지
#
# ablation 옵션 (ResyncOptions):
#   warmup_window   nothing = 전체 이력(기본). 정수 W = donor 이력의 최근 W step만 재생.
#                   **최근 W step 재생의 시작 state도 그 멤버의 현재 state가 아니라 학습 직후 state다**
#                   (입력 = y_{T−W}, …, y_{T−1}; W step이 washout 역할을 한다).
#   correction_mode :donor_history (기본) / :none (판정·기록만, 교정 안 함 → single_all과 같아야 함) /
#                   :prune (bad를 앙상블에서 영구 제거; 제거된 멤버는 이후 rollout·판정·donor에서 빠짐)
#   bad_selector    criterion 이름(Symbol, CRITERIA_ALL 중 하나) 또는 함수 (pool, ctx) -> scores
#   apply_to_all    true면 bad 판정 없이 활성 멤버 전원을 교정 (donor는 자기 자신을 뺀 활성 멤버 중 배정 규칙대로)
#
# 기록 (segment마다): bad (N×n_seg Bool), donor (N×n_seg Int, 0 = 교정 안 됨), ancestor (N×n_seg Int, 교정 후
#   각 멤버 궤적의 원조 멤버; 교정되면 donor의 원조를 상속), active (N×n_seg), n_corrections, last_corrected,
#   criterion 점수 (N×n_seg), 그리고 truth 기반 **평가용** segment 통계(선택에는 truth를 쓰지 않는다).
#
# 저장: run_dir/{chunk_XXX.jld2 (D × ckpt_every·τ × N, 멤버의 교정 반영 "유효" 궤적), state.jld2 (재개용),
#   result.jld2}. slot 저장소 = 유효 궤적: 교정된 j의 segment s slot은 donor 후보로 덮어쓰고, lineage 표
#   src[j, 1:s−1] ← src_old[d, 1:s−1] (이번 segment 교정 전 snapshot), src[j, s] = j.
#   StatementI는 src[j, s] = d였는데 slot j·s와 slot d·s의 내용이 같으므로 조립 결과는 bit 단위로 같고,
#   apply_to_all(donor도 같은 segment에 교정됨)에서도 일관된다.
# ============================================================

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const R_OUTPUT_ROOT = joinpath(REPO_ROOT, "outputs")          # every run writes under outputs/ (git-ignored)

include(joinpath(REPO_ROOT, "src", "prelude.jl"))
using .TestUtils
using Statistics, LinearAlgebra, Printf, Random, JLD2, FFTW, CairoMakie

include(joinpath(@__DIR__, "feature_extractors.jl"))
using .FeatureExtractors
include(joinpath(@__DIR__, "segment_metrics.jl"))                 # global_truth_variance / pointwise_nmse / segment_wasserstein
include(joinpath(REPO_ROOT, "src", "core", "training.jl"))        # train_ensemble_checkpointed
include(joinpath(REPO_ROOT, "src", "core", "robust_run.jl"))      # init_run_log / log_line! / heartbeat!
include(joinpath(@__DIR__, "reservoir_correction_engine.jl"))     # nearest_to_consensus
include(joinpath(@__DIR__, "reservoir_geometry_metrics.jl"))      # training_geometry_profile / score_against_profile
include(joinpath(@__DIR__, "blind_selection_engine.jl"))          # _rollout_with_states / series_moments

CairoMakie.activate!()

# ------------------------------------------------------------
# 0. 상수 / criterion 목록
# ------------------------------------------------------------

const VIABLE_CRITERIA = [:feature_distance_from_centroid, :phase_coverage_distance, :spectral_stat_to_train,
                         :stability_filter, :wasserstein_to_train, :reservoir_geometry_leverage]
const EXCLUDED_CRITERIA = [:tail_divergence, :local_lyapunov_proxy, :ensemble_outlier_score, :reservoir_geometry_subspace]
const DEFAULT_CRITERIA = VIABLE_CRITERIA
const CRITERIA_ALL = vcat(DEFAULT_CRITERIA, EXCLUDED_CRITERIA)
const GEOMETRY_CRITERIA = (:reservoir_geometry_leverage, :reservoir_geometry_subspace)

const PCA_VAR_TARGET = 0.95
const PHASE_NBINS = 15
const GEOMETRY_LAMBDA = 1e-6
const GEOMETRY_VARIANCE_THRESHOLD = 0.99
const DONOR_SEED_DEFAULT = 20260923

function atomic_save(path::AbstractString; kwargs...)
    tmp = path * ".tmp"
    jldsave(tmp; kwargs...)
    mv(tmp, path; force=true)
end

seg_range(s::Int, τ::Int) = ((s - 1) * τ + 1):(s * τ)
qtl(v, p) = (f = filter(isfinite, v); isempty(f) ? NaN : quantile(f, p))

# ------------------------------------------------------------
# 1. 시스템 spec
# ------------------------------------------------------------

"""
    duffing_system(; ...) -> NamedTuple

StatementI_resync.jl과 동일한 Duffing 설정 (duffing_oscillator.jld2, shift=1000, train_len=5000, 입력 noise 1e-6,
res_size=200, radius 0.9, sparsity 6/res, leak 0.3, ridge 1e-6, washout 500). R0 selftest와 R5(실데이터 spec)가 쓴다.
R1–R4 / R_system의 시스템 정의는 system_specs의 설정 (R0/run_analysis.jl r_system_setup → sys_from_config)이고,
건강도 판정기 보정(긴 truth)도 거기 (build_eval_env) 하나다.
"""
duffing_system(; kwargs...) = merge((
    name="duffing",
    data_path=joinpath(REPO_ROOT, "data", "duffing", "duffing_oscillator.jld2"),
    shift=1000, train_len=5000, noise=1e-6, noise_seed=1234,
    res_size=200, radius=0.9, sparsity=6 / 200, leak=0.3, ridge=1e-6, washout=500, input_scaling=0.1,
    state_dim=2, spatial=false, sign_regime_dim=1, regime_K=8,
), (; kwargs...))

# ------------------------------------------------------------
# 2. FE1 / training PCA / phase grid (StatementI와 동일; phase grid만 D>2 일반화)
# ------------------------------------------------------------

"X (D×T) → S (D×4), S[d,k] = Σ_t X[d,t]^k."
function power_sums(X::AbstractMatrix)
    D = size(X, 1)
    S = zeros(D, 4)
    @inbounds for t in axes(X, 2), d in 1:D
        x = X[d, t]
        x2 = x * x
        S[d, 1] += x; S[d, 2] += x2; S[d, 3] += x2 * x; S[d, 4] += x2 * x2
    end
    return S
end

"power sums(n 샘플) → FE1 [mean(D); std(D); skew(D); excess kurt(D)] (모집단 모멘트)."
function fe1_from_sums(S::AbstractMatrix, n::Real)
    D = size(S, 1)
    f = Vector{Float64}(undef, 4D)
    for d in 1:D
        m = S[d, 1] / n; e2 = S[d, 2] / n; e3 = S[d, 3] / n; e4 = S[d, 4] / n
        v = max(e2 - m^2, 0.0)
        mu3 = e3 - 3m * e2 + 2m^3
        mu4 = e4 - 4m * e3 + 6m^2 * e2 - 3m^4
        sd = sqrt(v)
        f[d] = m
        f[D + d] = sd
        f[2D + d] = v > 1e-14 ? mu3 / sd^3 : 0.0
        f[3D + d] = v > 1e-14 ? mu4 / v^2 - 3 : 0.0
    end
    return f
end
fe1(X::AbstractMatrix) = fe1_from_sums(power_sums(X), size(X, 2))

"training을 길이-L 창(stride)으로 잘라 FE1 → z-score → SVD PCA. k_crit = 누적 설명분산 ≥ 95% 최소 성분 수(≥2)."
function fit_train_pca(train::AbstractMatrix, L::Int, stride::Int)
    T = size(train, 2)
    F = reduce(hcat, [fe1(view(train, :, s:(s + L - 1))) for s in 1:stride:(T - L + 1)])
    mu = vec(mean(F; dims=2))
    sigma = vec(std(F; dims=2))
    for i in eachindex(sigma)
        (!isfinite(sigma[i]) || sigma[i] == 0) && (sigma[i] = 1.0)
    end
    Fs = svd(permutedims((F .- mu) ./ sigma))
    ev = Fs.S .^ 2
    ratio = ev ./ sum(ev)
    k_crit = max(2, something(findfirst(>=(PCA_VAR_TARGET), cumsum(ratio)), length(ratio)))
    return (mu=mu, sigma=sigma, V=Fs.V, ratio=ratio, k_crit=k_crit, n_windows=size(F, 2))
end
pca_project(pca, f::AbstractVector; k::Int=2) = pca.V[:, 1:k]' * ((f .- pca.mu) ./ pca.sigma)

"phase grid용 2D 투영: D == 2면 원 좌표 그대로(StatementI와 동일), D > 2면 training state의 상위 2 주성분."
function phase_projection(train::AbstractMatrix)
    D = size(train, 1)
    D == 2 && return (mu=zeros(2), P=Matrix{Float64}(I, 2, 2), identity=true)
    mu = vec(mean(train; dims=2))
    U = svd(train .- mu).U
    return (mu=mu, P=Matrix(U[:, 1:2]'), identity=false)
end
project2(pp, X::AbstractMatrix) = pp.identity ? X : pp.P * (X .- pp.mu)

function mark_cells!(B::AbstractMatrix{Bool}, X::AbstractMatrix, lo, hi, nb::Int)
    @inbounds for t in axes(X, 2)
        x1 = X[1, t]; x2 = X[2, t]
        (isfinite(x1) && isfinite(x2)) || continue
        (lo[1] <= x1 <= hi[1] && lo[2] <= x2 <= hi[2]) || continue
        i = clamp(floor(Int, (x1 - lo[1]) / (hi[1] - lo[1]) * nb) + 1, 1, nb)
        j = clamp(floor(Int, (x2 - lo[2]) / (hi[2] - lo[2]) * nb) + 1, 1, nb)
        B[i, j] = true
    end
    return B
end

function build_train_phase_grid(train::AbstractMatrix; nbins::Int=PHASE_NBINS, margin::Float64=0.05)
    pp = phase_projection(train)
    Z = project2(pp, train)
    lo = [minimum(view(Z, d, :)) for d in 1:2]
    hi = [maximum(view(Z, d, :)) for d in 1:2]
    w = hi .- lo
    lo .-= margin .* w
    hi .+= margin .* w
    cells = mark_cells!(fill(false, nbins, nbins), Z, lo, hi, nbins)
    return (pp=pp, lo=lo, hi=hi, nbins=nbins, train_cells=cells, n_train=count(cells))
end

"coverage = (궤적이 방문한 training-attractor 셀 수) / (training-attractor 셀 수)."
function phase_coverage(grid, X::AbstractMatrix)
    B = mark_cells!(fill(false, grid.nbins, grid.nbins), project2(grid.pp, X), grid.lo, grid.hi, grid.nbins)
    return count(B .& grid.train_cells) / grid.n_train
end

# ------------------------------------------------------------
# 3. training-reference helper (StatementC 정의; statement_I_engine.jl과 동일)
# ------------------------------------------------------------

function train_wasserstein_distance(train_ref::AbstractMatrix, candidate::AbstractMatrix; n_quantiles::Int=200)
    return mean(DynamicsMetrics.evaluate(
        DynamicsMetrics.QuantileWassersteinDistance(n_quantiles=n_quantiles, reduction=:state),
        train_ref, candidate).values)
end

function windowed_log_power_spectrum(data::AbstractMatrix, window_len::Int; hop::Int=max(window_len ÷ 2, 1))
    T = size(data, 2)
    starts = collect(1:hop:(T - window_len + 1))
    isempty(starts) && (starts = [1])
    acc = abs2.(rfft(view(data, :, starts[1]:(starts[1] + window_len - 1)), 2)) ./ window_len
    for s in starts[2:end]
        acc .+= abs2.(rfft(view(data, :, s:(s + window_len - 1)), 2)) ./ window_len
    end
    acc ./= length(starts)
    return log1p.(acc)
end

function spectral_distance_to_reference(candidate::AbstractMatrix, ref_log_psd::AbstractMatrix)
    cand_log_psd = log1p.(abs2.(rfft(candidate, 2)) ./ size(candidate, 2))
    return sqrt(mean(abs2, cand_log_psd .- ref_log_psd))
end

function spectral_stat_score(cand::AbstractMatrix, train_moments, ref_log_psd)
    m = series_moments(cand)
    mean_d = mean(abs.(m.mean .- train_moments.mean) ./ sqrt.(train_moments.var .+ eps()))
    var_d = mean(abs.(log.(m.var .+ eps()) .- log.(train_moments.var .+ eps())))
    skew_d = mean(abs.(m.skew .- train_moments.skew))
    kurt_d = mean(abs.(m.kurt .- train_moments.kurt))
    spec = spectral_distance_to_reference(cand, ref_log_psd)
    score = mean_d + var_d + 0.5 * skew_d + 0.25 * kurt_d + (isfinite(spec) ? spec : 0.0)
    return isfinite(score) ? score : Inf
end

function stability_score(cand::AbstractMatrix, train_moments, lo, hi)
    ok = all(vec(minimum(cand; dims=2)) .>= lo) && all(vec(maximum(cand; dims=2)) .<= hi)
    ok || return Inf
    cv = vec(var(cand; dims=2, corrected=false))
    return mean(abs.(log.(cv .+ eps()) .- log.(train_moments.var .+ eps())))
end

# ------------------------------------------------------------
# 4. 수치 helper (StatementI와 동일)
# ------------------------------------------------------------

function finite_mean(pool::AbstractVector, fin::AbstractVector{Bool})
    g = [pool[i] for i in eachindex(pool) if fin[i]]
    return isempty(g) ? fill(NaN, size(pool[1])) : sum(g) ./ length(g)
end

"Σ_{t∈rng} ‖P(:,t) − M(:,t)‖"
function divergence_sum(P::AbstractMatrix, M::AbstractMatrix, rng=axes(P, 2))
    s = 0.0
    @inbounds for t in rng
        d2 = 0.0
        for d in axes(P, 1)
            δ = P[d, t] - M[d, t]
            d2 += δ * δ
        end
        s += sqrt(d2)
    end
    return s
end

"Σ_t Σ_d (P − truth)² / variance_scale[d]"
function scaled_sqerr_sum(P::AbstractMatrix, truth_seg::AbstractMatrix, vs::AbstractVector)
    s = 0.0
    @inbounds for t in axes(P, 2), d in axes(P, 1)
        δ = P[d, t] - truth_seg[d, t]
        s += δ * δ / vs[d]
    end
    return s
end

"finite 멤버 i마다 acc[i] = Σ_{j≠i, finite} Σ_t ‖P_i(t) − P_j(t)‖, cnt = (m−1)·L."
function pairwise_distance_sums(pool::AbstractVector, fin::AbstractVector{Bool})
    N = length(pool)
    acc = zeros(N)
    idx = findall(fin)
    m = length(idx)
    m <= 1 && return acc, 0.0, m
    D, L = size(pool[idx[1]])
    Threads.@threads for a in 1:m
        Pi = pool[idx[a]]
        s = 0.0
        @inbounds for b in 1:m
            b == a && continue
            Pj = pool[idx[b]]
            for t in 1:L
                d2 = 0.0
                for d in 1:D
                    δ = Pi[d, t] - Pj[d, t]
                    d2 += δ * δ
                end
                s += sqrt(d2)
            end
        end
        acc[idx[a]] = s
    end
    return acc, Float64((m - 1) * L), m
end

# ------------------------------------------------------------
# 5. 재생 (teacher-forcing)
# ------------------------------------------------------------

"""
    warmup_full_history(ps, initial_state, esn, donor_traj; x0) -> new_state

`initial_state`(학습 직후 state)에서 donor 생성 이력 `donor_traj`(D × T)를 teacher-forcing.
입력 = hcat(x0, donor_traj[:, 1:end-1]); 호출자는 다음 자율 입력으로 donor_traj[:, end]를 쓴다.
"""
function warmup_full_history(ps, initial_state, esn, donor_traj::AbstractMatrix; x0::AbstractVector)
    inputs = hcat(x0, view(donor_traj, :, 1:(size(donor_traj, 2) - 1)))
    return TestUtils.warmup_teacher_forcing(ps, deepcopy(initial_state), esn, inputs)
end

"""
    warmup_recent(ps, initial_state, esn, donor_traj, W; x0) -> new_state

donor 이력의 최근 W step만 재생. **시작 state는 이 멤버의 현재 state가 아니라 학습 직후 state**이며,
입력 = donor_traj[:, T−W : T−1] (T = 이력 길이; W ≥ T면 warmup_full_history와 같다).
"""
function warmup_recent(ps, initial_state, esn, donor_traj::AbstractMatrix, W::Int; x0::AbstractVector)
    T = size(donor_traj, 2)
    W >= T && return warmup_full_history(ps, initial_state, esn, donor_traj; x0=x0)
    return TestUtils.warmup_teacher_forcing(ps, deepcopy(initial_state), esn, donor_traj[:, (T - W):(T - 1)])
end

"lineage `src`로 멤버 j의 segment 1..s 유효 이력을 slot 저장소에서 조립 (D × s·τ)."
function assemble_history(hist::AbstractVector, src::AbstractMatrix{Int}, j::Int, s::Int, τ::Int)
    D = size(hist[1], 1)
    Y = Matrix{Float64}(undef, D, s * τ)
    for k in 1:s
        r = seg_range(k, τ)
        Y[:, r] .= view(hist[src[j, k]], :, r)
    end
    return Y
end

# ------------------------------------------------------------
# 6. Context
# ------------------------------------------------------------

function compute_geometry_profiles(ensemble, train_input, washout, path, runlog)
    isfile(path) && return JLD2.load(path, "profiles")
    N = length(ensemble)
    profiles = Vector{Any}(undef, N)
    Threads.@threads for i in 1:N
        ps, st, esn = ensemble[i]
        states, _ = ReservoirComputing.collectstates(esn, train_input, ps, deepcopy(st))
        profiles[i] = training_geometry_profile(Matrix{Float64}(states[:, (washout + 1):end]);
            λ=GEOMETRY_LAMBDA, variance_threshold=GEOMETRY_VARIANCE_THRESHOLD)
    end
    atomic_save(path; profiles=profiles)
    log_line!(runlog, "geometry profiles computed ($N members)")
    return profiles
end

"""
    build_resync_context(sys; n_ensemble, τ, n_segments, checkpoint_dir, log_dir, log_name,
                         seed_start=1234, test_offset=0, sync_len=1000, need_geometry=true) -> ctx

- `test_offset` = 0이면 test가 training 직후에서 시작(StatementI와 동일). > 0이면 test 시작점을 그만큼 뒤로
  옮기고(다른 초기 조건), 각 멤버를 학습 직후 state에서 test 직전 `sync_len` step의 관측 데이터로
  teacher-forcing해 동기화한다. 이때 재생의 "학습 직후 state"(init_states)는 이 동기화된 state다.
- ensemble checkpoint는 `checkpoint_dir`(fingerprint 검사 포함); geometry profile은 그 옆에 캐시.
"""
function build_resync_context(sys; n_ensemble::Int, τ::Int, n_segments::Int, checkpoint_dir::AbstractString,
                              log_dir::AbstractString, log_name::AbstractString, seed_start::Int=1234,
                              test_offset::Int=0, sync_len::Int=1000, need_geometry::Bool=true, runlog=nothing,
                              geometry_path::Union{Nothing,AbstractString}=nothing)
    mkpath(log_dir)
    runlog = runlog === nothing ? init_run_log(log_dir, log_name) : runlog
    predict_len = τ * n_segments
    gen = get(sys, :generator, nothing)
    (!isfile(sys.data_path) && gen !== nothing) && gen(sys.data_path)          # 데이터가 없으면 생성
    data = TestUtils.load_dataset(data_path=sys.data_path, shift=sys.shift, train_len=sys.train_len,
        valid_len=test_offset, predict_len=predict_len)
    # normalize=true: training target의 차원별 (μ, σ)로 전체를 표준화 (truth 통계는 쓰지 않는다)
    nmu, nsd = get(sys, :normalize, false) ? (vec(mean(data.train_target; dims=2)), vec(std(data.train_target; dims=2)) .+ eps()) :
               (zeros(size(data.train_target, 1)), ones(size(data.train_target, 1)))
    nz(X) = (Matrix{Float64}(X) .- nmu) ./ nsd
    data = merge(data, (sol=nz(data.sol), test_target=nz(data.test_target), train_input=nz(data.train_input),
                        train_target=nz(data.train_target), initial_data=(Vector{Float64}(data.initial_data) .- nmu) ./ nsd))
    truth = Matrix{Float64}(data.test_target)
    x0 = Vector{Float64}(collect(data.initial_data))
    train_input = Matrix{Float64}(data.train_input)
    train_target = Matrix{Float64}(data.train_target)
    TestUtils.add_noise!(train_input; noiselevel=sys.noise, rng=MersenneTwister(sys.noise_seed))

    base_config = TestUtils.default_experiment_config(
        in_size=sys.state_dim, res_size=sys.res_size, sparsity=sys.sparsity, radius=sys.radius,
        leak_coefficient=sys.leak, out_size=sys.state_dim, scaling=sys.input_scaling,
        training_method=StandardRidge(sys.ridge), solver=QRSolver(), washout=sys.washout,
        bias_scaling=Float64(get(sys, :bias_scaling, 0.0)), state_modifiers=TestUtils.readout_modifiers(get(sys, :readout, "linear")))
    ensemble = train_ensemble_checkpointed(train_input, train_target, n_ensemble;
        base_config=base_config, vary_seed=true, seed_start=seed_start, checkpoint_dir=checkpoint_dir,
        sparse=Bool(get(sys, :sparse, false)),
        on_progress=(i, n) -> (i % 50 == 0 || i == n) && log_line!(runlog, "ensemble $i/$n"))
    log_line!(runlog, "[$(sys.name)] ensemble ready ($(length(ensemble)), seed_start=$seed_start)")

    init_states = [deepcopy(e[2]) for e in ensemble]
    if test_offset > 0
        tstart = sys.shift + sys.train_len + test_offset       # sol 열 index of initial_data
        sl = min(sync_len, test_offset)
        sync_in = Matrix{Float64}(data.sol[:, (tstart - sl):(tstart - 1)])
        Threads.@threads for i in eachindex(ensemble)
            ps, st, esn = ensemble[i]
            init_states[i] = TestUtils.warmup_teacher_forcing(ps, deepcopy(st), esn, sync_in)
        end
        ensemble = [(e[1], deepcopy(init_states[i]), e[3]) for (i, e) in enumerate(ensemble)]
        log_line!(runlog, "[$(sys.name)] test_offset=$test_offset: members synced on $sl observed steps")
    end

    train_moments = series_moments(train_target)
    margin = 4.0 .* sqrt.(train_moments.var)
    gpath = geometry_path !== nothing ? geometry_path :
        joinpath(dirname(normpath(checkpoint_dir)), basename(normpath(checkpoint_dir)) * "_geometry_profiles_n$(n_ensemble)" *
                 (test_offset > 0 ? "_off$(test_offset)" : "") * ".jld2")
    geom = need_geometry ? compute_geometry_profiles(ensemble, train_input, sys.washout, gpath, runlog) : nothing
    return (sys=sys, ensemble=ensemble, n=length(ensemble), init_states=init_states, x0=x0, norm=(mu=nmu, sd=nsd),
        truth=truth, variance_scale=global_truth_variance(truth), τ=τ, n_segments=n_segments, predict_len=predict_len,
        train_input=train_input, train_target=train_target,
        pca=fit_train_pca(train_target, τ, max(1, τ ÷ 10)), phase_grid=build_train_phase_grid(train_target),
        train_moments=train_moments, stab_lo=vec(minimum(train_target; dims=2)) .- margin,
        stab_hi=vec(maximum(train_target; dims=2)) .+ margin, ref_log_psd=windowed_log_power_spectrum(train_target, τ),
        geom_profiles=geom, test_offset=test_offset, seed_start=seed_start, runlog=runlog)
end

"""
    replicate_specs(n_ic, n_seed; ic_step=150_000) -> Vector{NamedTuple}

R2~R4 반복 설정: test 초기 조건 j → test_offset = (j−1)·ic_step (j=1은 StatementI와 같은 test 시작),
ensemble seed set k → seed_start = 1234 + 100_000·(k−1). ic_step = 150000 step이면 predict_len 120000인 test
구간끼리 겹치지 않는다.
"""
replicate_specs(n_ic::Int, n_seed::Int; ic_step::Int=150_000) =
    [(ic=j, seed=k, test_offset=(j - 1) * ic_step, seed_start=1234 + 100_000 * (k - 1), tag="ic$(j)_s$(k)")
     for k in 1:n_seed for j in 1:n_ic]
"앙상블 checkpoint 디렉터리. sys.ens_name이 있으면 그 이름을 쓴다 (예: smoke 전용 축소 설정을 full과 분리)."
ensemble_dir(sys, seed_start::Int) = joinpath(R_OUTPUT_ROOT, "ensembles", "$(get(sys, :ens_name, sys.name))_seed$(seed_start)")

"같은 앙상블로 멤버 subset만 쓰는 context (bad apple 제거 / 사전 선별). `keep`은 이 ctx 안의 멤버 번호."
subset_context(ctx, keep::AbstractVector{Int}) = merge(ctx, (ensemble=ctx.ensemble[keep], n=length(keep),
    init_states=ctx.init_states[keep], geom_profiles=ctx.geom_profiles === nothing ? nothing : ctx.geom_profiles[keep],
    member_ids=collect(get(ctx, :member_ids, 1:ctx.n))[keep]))

# ------------------------------------------------------------
# 6b. 기본 baseline: segment-1 refill until clean (논문 결정 2026-10-11; StatementI_refill.jl 2b 규칙)
# ------------------------------------------------------------

"""
    refill_until_clean(ctx_all, n, crit; p=0.10, max_rounds=50, random_seed=777) -> (ids, info)

ctx_all = 원래 n개 + 예비 멤버 (seed 순서; 같은 checkpoint 디렉터리의 n+1번째부터). truth는 쓰지 않는다.
1. 모든 멤버의 segment-1 후보 (학습 직후 state에서 τ step 자율 rollout).
2. 원래 n개 pool에 criterion → K = round(p·n)번째로 큰 점수가 고정 cutoff.
3. 지금 pool의 점수(pool에 의존하는 criterion이라 매 round 다시 계산)가 cutoff 이상이거나 비유한인 멤버를 다음 예비
   멤버로 교체, 없어질 때까지 반복 (round 1 = 2a "한 번 교체").
반환 ids = ctx_all 안의 멤버 번호 (길이 n), info = (crit, p, cutoff, converged, rounds, n_new, reserve_used).
"""
function refill_until_clean(ctx_all, n::Int, crit::Symbol; p::Real=0.10, max_rounds::Int=50, random_seed::Int=777)
    M = ctx_all.n
    @assert M > n "refill needs reserve members (ctx_all.n = $M, n = $n)"
    need_geom = crit in GEOMETRY_CRITERIA
    need_geom && @assert ctx_all.geom_profiles !== nothing "refill criterion $crit needs geometry profiles"
    cands = Vector{Matrix{Float64}}(undef, M); st = Vector{Any}(undef, M); geom = fill(NaN, M, 2)
    generate_candidates!(cands, st, geom, ctx_all, Any[deepcopy(e[2]) for e in ctx_all.ensemble], [copy(ctx_all.x0) for _ in 1:M],
        fill(true, M), need_geom)
    fin = [all(isfinite, c) for c in cands]
    K = clamp(round(Int, n * p), 1, n - 1)
    score(ids) = criterion_scores(crit, SegmentPool(cands[ids], fin[ids], geom[ids, :], 1, random_seed), ctx_all)
    S0 = score(collect(1:n))
    cutoff = S0[sortperm(S0; rev=true)[K]]           # StatementI_refill 2b와 같은 규칙 (비유한 점수 Inf 포함 K번째)
    ids = collect(1:n); ptr = 0; rounds = NamedTuple[]; converged = false
    for r in 1:max_rounds
        S = score(ids)
        flagged = [a for a in 1:n if !fin[ids[a]] || S[a] >= cutoff]
        isempty(flagged) && (converged = true; break)
        n + ptr + length(flagged) <= M || error("refill exhausted the reserve (round $r, used $ptr of $(M - n), need $(length(flagged)) more) -- raise the reserve")
        removed = ids[flagged]
        added = collect((n + ptr + 1):(n + ptr + length(flagged)))
        ids[flagged] .= added
        ptr += length(flagged)
        push!(rounds, (round=r, removed=removed, added=added))
    end
    return ids, (crit=crit, p=p, cutoff=cutoff, converged=converged, rounds=rounds, n_new=count(>(n), ids), reserve_used=ptr)
end

"""
    build_refilled_context(sys; n_ensemble, reserve, refill_criterion, refill_p, checkpoint_dir, kwargs...) -> ctx

R 실험의 기본 앙상블: n + reserve 멤버로 build_resync_context(같은 checkpoint 디렉터리 = 같은 seed 순서; 앞 n개는 원래
앙상블과 같다) → refill_until_clean → 그 n개의 subset context. baseline = 이 앙상블의 free-running(single_all_options),
framework = 같은 앙상블 위의 resync (resync만의 효과). refill 결과는 checkpoint 디렉터리에 캐시 (refill_<hash>.jld2).
ctx.refill = info (+ ids), ctx.member_ids = 전역 멤버 번호 (> n_ensemble = 예비에서 온 새 seed; run fingerprint에 들어간다).
기본: reserve = sys.refill_reserve 또는 ⌈n/2⌉, criterion = sys.refill_criterion 또는 :feature_distance_from_centroid, p = 0.10.
"""
function build_refilled_context(sys; n_ensemble::Int, checkpoint_dir::AbstractString,
                                reserve::Int=get(sys, :refill_reserve, cld(n_ensemble, 2)),
                                refill_criterion::Symbol=get(sys, :refill_criterion, :feature_distance_from_centroid),
                                refill_p::Real=0.10, kwargs...)
    ctx_all = build_resync_context(sys; n_ensemble=n_ensemble + reserve, checkpoint_dir=checkpoint_dir, kwargs...)
    fp = (n_ensemble, reserve, refill_criterion, refill_p, ctx_all.τ, ctx_all.test_offset, ctx_all.seed_start, ctx_all.x0, sys.name)
    path = joinpath(checkpoint_dir, "refill_$(string(hash(fp); base=16)).jld2")
    if isfile(path) && JLD2.load(path, "fp") == fp
        ids = JLD2.load(path, "ids"); info = JLD2.load(path, "info")
    else
        ids, info = refill_until_clean(ctx_all, n_ensemble, refill_criterion; p=refill_p)
        atomic_save(path; ids=ids, info=info, fp=fp)
    end
    log_line!(ctx_all.runlog, @sprintf("[%s] refill until clean (%s, p=%.2f): %d round(s) %s, %d new-seed members (reserve used %d/%d)",
        sys.name, refill_criterion, refill_p, length(info.rounds), info.converged ? "converged" : "NOT converged", info.n_new,
        info.reserve_used, reserve))
    ctx = subset_context(merge(ctx_all, (member_ids=collect(1:ctx_all.n),)), ids)
    return merge(ctx, (refill=merge(info, (ids=ids,)), n_original=n_ensemble))
end

# ------------------------------------------------------------
# 7. Bad criteria (truth 미사용; "값이 클수록 bad"; 비유한 = Inf)
# ------------------------------------------------------------

"segment 하나의 후보 pool (criterion 공용 입력). fin = 유한 & 활성."
struct SegmentPool
    cands::Vector{Matrix{Float64}}
    fin::Vector{Bool}
    geom::Matrix{Float64}          # N × 2 (leverage, subspace); 미계산 시 NaN
    s::Int
    seed::Int
end

"""
feature_distance_from_centroid — 보는 것: 멤버 궤적의 FE1(차원별 mean/std/skew/excess-kurt)을 training-고정 PCA
(누적 설명분산 95% 성분)로 투영했을 때, 같은 segment의 다른 멤버들 FE1 중심에서 얼마나 먼가. 참조 training
통계: 길이-τ training 창들의 FE1 평균·표준편차와 PCA 기저 (`ctx.pca`).
"""
function crit_feature_distance_from_centroid(pool::SegmentPool, ctx)
    N = length(pool.cands); s = fill(Inf, N)
    k = ctx.pca.k_crit
    z = [pool.fin[i] ? pca_project(ctx.pca, fe1(pool.cands[i]); k=k) : fill(NaN, k) for i in 1:N]
    any(pool.fin) || return s
    c = mean(z[i] for i in 1:N if pool.fin[i])
    for i in 1:N
        pool.fin[i] && (s[i] = norm(z[i] .- c))
    end
    return s
end

"FE1 PCA (at most 8 axes): distance from the coordinate-wise median of finite members."
function crit_feature_distance_from_median8(pool::SegmentPool, ctx)
    N = length(pool.cands); s = fill(Inf, N)
    k = min(ctx.pca.k_crit, 8)
    z = [pool.fin[i] ? pca_project(ctx.pca, fe1(pool.cands[i]); k=k) : fill(NaN, k) for i in 1:N]
    ids = [i for i in 1:N if all(isfinite, z[i])]
    isempty(ids) && return s
    c = [median(z[i][d] for i in ids) for d in 1:k]
    for i in ids
        s[i] = norm(z[i] .- c)
    end
    return s
end

"FE1 PCA (at most 8 axes): local outlier factor with 5 neighbours."
function crit_feature_lof5_8(pool::SegmentPool, ctx)
    N = length(pool.cands); s = fill(Inf, N)
    kdim = min(ctx.pca.k_crit, 8)
    z = [pool.fin[i] ? pca_project(ctx.pca, fe1(pool.cands[i]); k=kdim) : fill(NaN, kdim) for i in 1:N]
    ids = [i for i in 1:N if all(isfinite, z[i])]
    length(ids) <= 1 && return s
    k = min(5, length(ids) - 1)
    dist = [i == j ? 0.0 : norm(z[i] .- z[j]) for i in ids, j in ids]
    neighbors = [[b for b in partialsortperm(view(dist, a, :), 1:(k + 1)) if b != a][1:k]
                 for a in eachindex(ids)]
    kth = [maximum(dist[a, b] for b in neighbors[a]) for a in eachindex(ids)]
    lrd = [1 / max(mean(max(kth[b], dist[a, b]) for b in neighbors[a]), eps()) for a in eachindex(ids)]
    for a in eachindex(ids)
        s[ids[a]] = mean(lrd[b] for b in neighbors[a]) / lrd[a]
    end
    return s
end

"""
phase_coverage_distance — 보는 것: 1 − (training attractor가 방문한 격자 셀 중 멤버가 방문한 비율). 고정점·한
well·축소된 궤도처럼 attractor를 적게 덮는 궤적. 참조 training 통계: training state의 15×15 격자 점유
(D = 2면 원 좌표, D > 2면 training 상위 2 주성분 평면; `ctx.phase_grid`).
"""
function crit_phase_coverage_distance(pool::SegmentPool, ctx)
    s = fill(Inf, length(pool.cands))
    for i in eachindex(s)
        pool.fin[i] && (s[i] = 1.0 - phase_coverage(ctx.phase_grid, pool.cands[i]))
    end
    return s
end

"""
spectral_stat_to_train — 보는 것: 멤버 궤적의 차원별 mean/var/skew/kurt가 training과 얼마나 다른가 + 멤버 log
PSD와 training의 창-평균 log PSD의 RMS 차이 (StatementC 점수 그대로). 참조 training 통계: `series_moments(train)`,
길이-τ 창 평균 log PSD (`ctx.ref_log_psd`).
"""
function crit_spectral_stat_to_train(pool::SegmentPool, ctx)
    s = fill(Inf, length(pool.cands))
    Threads.@threads for i in eachindex(s)
        pool.fin[i] && (s[i] = spectral_stat_score(pool.cands[i], ctx.train_moments, ctx.ref_log_psd))
    end
    return s
end

"""
stability_filter — 보는 것: 궤적이 training 범위 ±4σ를 벗어나면 Inf(최악), 아니면 차원별 log-variance의
training 대비 차이 (StatementC 점수 그대로). 발산과 진폭 붕괴를 가장 직접적으로 잡는다. 참조 training 통계:
차원별 min/max/분산.
"""
function crit_stability_filter(pool::SegmentPool, ctx)
    s = fill(Inf, length(pool.cands))
    for i in eachindex(s)
        pool.fin[i] && (s[i] = stability_score(pool.cands[i], ctx.train_moments, ctx.stab_lo, ctx.stab_hi))
    end
    return s
end

"""
wasserstein_to_train — 보는 것: 멤버 궤적 값 분포(차원별 주변 분포)와 training 값 분포의 quantile-Wasserstein
거리 (200 quantile; StatementC). 참조 training 통계: training target 전체의 차원별 quantile.
"""
function crit_wasserstein_to_train(pool::SegmentPool, ctx)
    s = fill(Inf, length(pool.cands))
    Threads.@threads for i in eachindex(s)
        pool.fin[i] && (s[i] = train_wasserstein_distance(ctx.train_target, pool.cands[i]))
    end
    return s
end

"""
reservoir_geometry_leverage — 보는 것: 멤버 **자신의** reservoir hidden state 궤적(이 segment)의 ridge-leverage
점수 분포가 그 멤버의 training hidden-state geometry와 얼마나 다른가 (quantile-W1; StatementC). 출력 궤적이
아니라 내부 state를 본다. 참조 training 통계: 멤버별 training hidden-state의 z-score·SVD·leverage quantile
(`ctx.geom_profiles`).
"""
crit_reservoir_geometry_leverage(pool::SegmentPool, ctx) =
    [pool.fin[i] ? pool.geom[i, 1] : Inf for i in eachindex(pool.cands)]

"""
(제외) tail_divergence — 보는 것: segment 후반 [3τ/4, τ]에서 앙상블 평균 궤적과의 거리. training 참조 없음.
"""
function crit_tail_divergence(pool::SegmentPool, ctx)
    N = length(pool.cands); s = fill(Inf, N)
    L = size(pool.cands[1], 2)
    M = finite_mean(pool.cands, pool.fin)
    r = max(1, 3L ÷ 4):L
    for i in 1:N
        pool.fin[i] && (s[i] = divergence_sum(pool.cands[i], M, r) / length(r))
    end
    return s
end

"""
(제외) local_lyapunov_proxy — 보는 것: 인접 step 변위 크기의 로그 비율 평균(국소 팽창률). training 참조 없음.
"""
function crit_local_lyapunov_proxy(pool::SegmentPool, ctx)
    N = length(pool.cands); s = fill(Inf, N)
    L = size(pool.cands[1], 2)
    for i in 1:N
        pool.fin[i] || continue
        X = pool.cands[i]
        acc = 0.0; cnt = 0
        for t in 2:(L - 1)
            num = norm(view(X, :, t + 1) .- view(X, :, t))
            den = norm(view(X, :, t) .- view(X, :, t - 1))
            if den > 1e-12
                acc += log(max(num, 1e-12) / den)
                cnt += 1
            end
        end
        s[i] = cnt == 0 ? Inf : acc / cnt
    end
    return s
end

"""
(제외) ensemble_outlier_score — 보는 것: 다른 모든 멤버와의 평균 pairwise 거리. training 참조 없음.
"""
function crit_ensemble_outlier_score(pool::SegmentPool, ctx)
    N = length(pool.cands); s = fill(Inf, N)
    acc, cnt, m = pairwise_distance_sums(pool.cands, pool.fin)
    if m > 1
        for i in 1:N
            pool.fin[i] && (s[i] = acc[i] / cnt)
        end
    end
    return s
end

"""
(제외) reservoir_geometry_subspace — 보는 것: 멤버 hidden state가 자기 training 주성분 subspace(99% 분산) 밖에
남기는 잔차 에너지. 참조 training 통계: 멤버별 training hidden-state SVD.
"""
crit_reservoir_geometry_subspace(pool::SegmentPool, ctx) =
    [pool.fin[i] ? pool.geom[i, 2] : Inf for i in eachindex(pool.cands)]

include(joinpath(@__DIR__, "experimental_selectors.jl"))

const CRITERION_FUNCS = Dict{Symbol,Function}(
    :trial_repair => crit_trial_repair,
    :peer_check => crit_peer_check,
    :distribution_influence => crit_distribution_influence,
    :feature_distance_from_centroid => crit_feature_distance_from_centroid,
    :feature_distance_from_median8 => crit_feature_distance_from_median8,
    :feature_lof5_8 => crit_feature_lof5_8,
    :phase_coverage_distance => crit_phase_coverage_distance,
    :spectral_stat_to_train => crit_spectral_stat_to_train,
    :stability_filter => crit_stability_filter,
    :wasserstein_to_train => crit_wasserstein_to_train,
    :reservoir_geometry_leverage => crit_reservoir_geometry_leverage,
    :tail_divergence => crit_tail_divergence,
    :local_lyapunov_proxy => crit_local_lyapunov_proxy,
    :ensemble_outlier_score => crit_ensemble_outlier_score,
    :reservoir_geometry_subspace => crit_reservoir_geometry_subspace,
)

"criterion(Symbol 또는 함수) 점수 (클수록 bad, NaN → Inf)."
function criterion_scores(crit, pool::SegmentPool, ctx)
    f = crit isa Symbol ? CRITERION_FUNCS[crit] : crit
    s = Vector{Float64}(f(pool, ctx))
    replace!(s, NaN => Inf)
    return s
end

"활성 멤버 중 score 상위 K개 + 비유한 활성 멤버 → 정렬된 bad index (전원 활성이면 StatementI select_bad와 동일)."
function select_bad(score::AbstractVector{<:Real}, fin::AbstractVector{Bool}, active::AbstractVector{Bool}, K::Int)
    act = findall(active)
    order = act[sortperm(score[act]; rev=true)]
    bad = Set(order[1:min(K, length(order))])
    for i in act
        fin[i] || push!(bad, i)
    end
    return sort(collect(bad))
end

# ------------------------------------------------------------
# 8. segment별 평가 통계 (truth 사용 = 평가 전용; 선택에는 안 씀. StatementI와 동일)
# ------------------------------------------------------------

function new_stats(N::Int, D::Int, nseg::Int)
    return (fin=fill(false, N, nseg), sqerr=fill(NaN, N, nseg), divsum=fill(NaN, N, nseg),
        right=fill(NaN, N, nseg), psum=fill(NaN, D, 4, N, nseg), pairsum=zeros(N, nseg),
        paircnt=zeros(nseg), nfin=zeros(Int, nseg),
        rep_idx=zeros(Int, nseg), rep_sqerr=fill(NaN, nseg), rep_div=fill(NaN, nseg),
        rep_pairsum=fill(NaN, nseg), rep_right=fill(NaN, nseg))
end

"`pool`의 비활성 멤버는 fin=false로 기록 (pairwise·평균에서도 빠진다)."
function record_segment_stats!(S, s::Int, pool::AbstractVector, truth_seg::AbstractMatrix, rep, ctx,
                               active::AbstractVector{Bool}=fill(true, length(pool)))
    N = length(pool)
    fin = [active[i] && all(isfinite, pool[i]) for i in 1:N]
    M = finite_mean(pool, fin)
    vs = ctx.variance_scale
    Threads.@threads for i in 1:N
        S.fin[i, s] = fin[i]
        fin[i] || continue
        P = pool[i]
        S.sqerr[i, s] = scaled_sqerr_sum(P, truth_seg, vs)
        S.divsum[i, s] = divergence_sum(P, M)
        S.right[i, s] = count(>(0), view(P, 1, :))
        S.psum[:, :, i, s] .= power_sums(P)
    end
    acc, cnt, m = pairwise_distance_sums(pool, fin)
    S.pairsum[:, s] .= acc
    S.paircnt[s] = cnt
    S.nfin[s] = m
    if rep !== nothing
        S.rep_idx[s] = rep
        S.rep_sqerr[s] = S.sqerr[rep, s]
        S.rep_div[s] = S.divsum[rep, s]
        S.rep_pairsum[s] = S.pairsum[rep, s]
        S.rep_right[s] = S.right[rep, s]
    end
    return S
end

"누적(1:c·τ) 멤버별 NMSE (가법 prefix sum; 비유한 segment가 있으면 NaN)."
function cumulative_nmse(S, c::Int, τ::Int)
    D = size(S.psum, 1)
    finall = vec(all(view(S.fin, :, 1:c); dims=2))
    v = vec(sum(view(S.sqerr, :, 1:c); dims=2)) ./ (c * τ * D)
    v[.!finall] .= NaN
    return v
end

# ------------------------------------------------------------
# 9. Rollout 엔진
# ------------------------------------------------------------

"""
    ResyncOptions(; p=0.10, donor=:uniform, warmup_window=nothing, correction_mode=:donor_history,
                   bad_selector=:stability_filter, apply_to_all=false, record_criteria=Symbol[],
                   ckpt_every=10, donor_seed=20260923, random_seed=777, min_active=2)

`record_criteria`: 매 segment 전원 점수를 추가로 기록할 criterion (single_all에서 R1b/R2c용).
`min_active`: :prune에서 활성 멤버가 이 수 아래로 내려가지 않게 제거를 멈춘다.
"""
Base.@kwdef struct ResyncOptions
    p::Float64 = 0.10
    donor::Symbol = :uniform
    warmup_window::Union{Nothing,Int} = nothing
    warmup_drop_segments::Int = 0
    correction_mode::Symbol = :donor_history
    bad_selector::Any = :stability_filter
    apply_to_all::Bool = false
    record_criteria::Vector{Symbol} = Symbol[]
    ckpt_every::Int = 10
    donor_seed::Int = DONOR_SEED_DEFAULT
    random_seed::Int = 777
    min_active::Int = 2
end

"Replay full history, a fixed recent window, or full history minus earliest segments (keep at least one)."
function correction_warmup(ctx, opts, j, Y)
    opts.warmup_drop_segments >= 0 || error("warmup_drop_segments must be nonnegative")
    opts.warmup_drop_segments > 0 && opts.warmup_window !== nothing &&
        error("choose warmup_window or warmup_drop_segments")
    ps, _, esn = ctx.ensemble[j]
    W = opts.warmup_drop_segments > 0 ? max(ctx.τ, size(Y, 2) - opts.warmup_drop_segments * ctx.τ) : opts.warmup_window
    return W === nothing ? warmup_full_history(ps, ctx.init_states[j], esn, Y; x0=ctx.x0) :
        warmup_recent(ps, ctx.init_states[j], esn, Y, W; x0=ctx.x0)
end

"single_all 설정: 교정 없음, 판정 없음, `record_criteria` 점수만 기록."
single_all_options(; record_criteria=DEFAULT_CRITERIA, ckpt_every::Int=10, random_seed::Int=777) =
    ResyncOptions(correction_mode=:none, bad_selector=nothing, record_criteria=collect(record_criteria),
                  ckpt_every=ckpt_every, random_seed=random_seed)

chunk_path(dir::AbstractString, c::Int) = joinpath(dir, @sprintf("chunk_%03d.jld2", c))
"""
chunk c의 "자기 출력" 기록 (교정 모드에서만): segment마다 교정된(= slot이 donor 후보로 덮어쓰인) 멤버와 덮어쓰기 전
그 멤버 자신의 autonomous candidate. `segs::Vector{Int}`, `bad::Vector{Vector{Int}}`, `cand::Vector{Array{Float64,3}}`
(D × τ × length(bad[k])). 유효 기록에서 이 칸만 바꾸면 멤버 자신의 출력 기록 (`load_member_trajectories(...; own=true)`).
archive/statement_verification/statement_I_engine.jl의 own_XXX.jld2와 같은 형식.
"""
own_path(dir::AbstractString, c::Int) = joinpath(dir, @sprintf("own_%03d.jld2", c))

function options_fingerprint(opts::ResyncOptions, ctx)
    sel = opts.bad_selector isa Function ? string(nameof(opts.bad_selector)) : string(opts.bad_selector)
    fp = (p=opts.p, donor=opts.donor, W=opts.warmup_window, mode=opts.correction_mode, sel=sel,
        all=opts.apply_to_all, rec=Tuple(opts.record_criteria), ck=opts.ckpt_every, ds=opts.donor_seed,
        rs=opts.random_seed, ma=opts.min_active, n=ctx.n, τ=ctx.τ, nseg=ctx.n_segments, off=ctx.test_offset,
        seed=ctx.seed_start, sys=ctx.sys.name, members=hash(get(ctx, :member_ids, nothing)))
    opts.warmup_drop_segments > 0 && (fp = merge(fp, (drop_segments=opts.warmup_drop_segments,)))
    haskey(ctx, :ensemble_signature) && (fp = merge(fp, (ensemble_signature=ctx.ensemble_signature,)))
    return fp
end

"저장된 run의 옵션 fingerprint가 지금과 같은가 (요약 캐시를 run 없이 읽기 전에 확인)."
run_matches(dir::AbstractString, opts::ResyncOptions, ctx) = (p = joinpath(dir, "result.jld2");
    isfile(p) && JLD2.load(p, "result").opts_fp == options_fingerprint(opts, ctx))

"N개 멤버 τ-step 자율 rollout (비활성 멤버는 NaN 후보). need_geom이면 hidden state로 geometry 점수도 계산."
function generate_candidates!(cands, st_new, geom, ctx, carried, next_inputs, active, need_geom::Bool)
    τ = ctx.τ
    D = size(ctx.truth, 1)
    Threads.@threads for i in 1:length(cands)
        if !active[i]
            cands[i] = fill(NaN, D, τ); st_new[i] = carried[i]; geom[i, :] .= Inf
            continue
        end
        ps, _, esn = ctx.ensemble[i]
        if need_geom
            out, states, stn = _rollout_with_states(esn, τ, ps, carried[i]; initialdata=next_inputs[i])
            cands[i] = Matrix{Float64}(out)
            st_new[i] = stn
            if ctx.geom_profiles !== nothing && all(isfinite, out) && all(isfinite, states)
                sc = score_against_profile(ctx.geom_profiles[i], Matrix{Float64}(states))
                geom[i, 1] = sc.leverage_distance
                geom[i, 2] = sc.subspace_residual
            else
                geom[i, :] .= Inf
            end
        else
            c, stn = ReservoirComputing.predict(esn, τ, ps, carried[i]; initialdata=next_inputs[i])
            cands[i] = Matrix{Float64}(c)
            st_new[i] = stn
        end
    end
end

"donor 배정: bad j → good pool 안의 donor (StatementI 규칙). apply_to_all이면 pool = 자기 자신을 뺀 활성 멤버."
function assign_donors(bad::Vector{Int}, good::Vector{Int}, opts::ResyncOptions, s::Int)
    donor = Dict{Int,Int}()
    drng = MersenneTwister(hash((opts.donor_seed, s)))
    for j in bad
        g = opts.apply_to_all ? [a for a in good if a != j] : good
        isempty(g) && continue
        ng = length(g)
        donor[j] = opts.donor === :uniform ? g[mod(j, ng) + 1] : g[rand(drng, 1:ng)]
    end
    return donor
end

"""
    run_resync(ctx, opts, dir; label="") -> result

`dir`에 chunk/state/result 저장, 재개 가능. `lineage_test_members`를 주면 끝에서 그 멤버들의 state를 lineage·slot
으로 재구성해 실제 carried state와 비교한다 (result.lineage_test = 최대 오차). `stop_after`(테스트용)는 그
segment의 checkpoint 저장 직후 `nothing`을 반환(중단 흉내). result 필드:
  bad, donor, ancestor, active, scores (자기 criterion, N×n_seg), recorded (Dict crit => N×n_seg 점수),
  n_corrections, last_corrected, K (segment별), stats (평가 통계), output (대표 = consensus 최근접 멤버),
  opts_fp, elapsed_sec.
"""
function run_resync(ctx, opts::ResyncOptions, dir::AbstractString; label::AbstractString=basename(dir),
                    lineage_test_members::AbstractVector{Int}=Int[], stop_after::Union{Nothing,Int}=nothing)
    mkpath(dir)
    result_path = joinpath(dir, "result.jld2")
    fp = options_fingerprint(opts, ctx)
    if isfile(result_path)
        r = JLD2.load(result_path, "result")
        r.opts_fp == fp && return r
        error("$result_path was produced with different options:\n  cached $(r.opts_fp)\n  now    $fp")
    end
    @assert opts.correction_mode in (:donor_history, :none, :prune) "unknown correction_mode $(opts.correction_mode)"
    @assert opts.donor in (:uniform, :random)
    state_path = joinpath(dir, "state.jld2")
    N = ctx.n; τ = ctx.τ; nseg = ctx.n_segments; D = size(ctx.truth, 1)
    ck = opts.ckpt_every
    @assert nseg % ck == 0 "n_segments must be a multiple of ckpt_every"
    judge = opts.bad_selector !== nothing || opts.apply_to_all
    correcting = judge && opts.correction_mode === :donor_history
    need_geom = ctx.geom_profiles !== nothing &&
        (opts.bad_selector in GEOMETRY_CRITERIA || any(c -> c in GEOMETRY_CRITERIA, opts.record_criteria))
    block_len = ck * τ
    predict_len = τ * nseg
    hist_len = correcting ? predict_len : block_len
    hist = [Matrix{Float64}(undef, D, hist_len) for _ in 1:N]
    hcols(r) = correcting ? r : (mod1(first(r), block_len):mod1(last(r), block_len))

    if isfile(state_path)
        st = JLD2.load(state_path, "state")
        @assert st.opts_fp == fp "state.jld2 options mismatch in $dir"
        s0 = st.seg + 1
        S = st.stats; output = st.output; carried = st.carried; next_inputs = st.next_inputs; src = st.src
        bad_m = st.bad; donor_m = st.donor; anc_m = st.ancestor; active_m = st.active; scores_m = st.scores
        recorded = st.recorded; ncorr = st.ncorr; lastc = st.lastc; anc = st.anc; active = st.active_now; Ks = st.K
        if correcting
            for c in 1:(st.seg ÷ ck)
                blk = JLD2.load(chunk_path(dir, c), "block")
                cols = ((c - 1) * block_len + 1):(c * block_len)
                for i in 1:N
                    hist[i][:, cols] .= view(blk, :, :, i)
                end
            end
        end
        log_line!(ctx.runlog, "[$label] resumed from segment $s0/$nseg")
    else
        s0 = 1
        S = new_stats(N, D, nseg)
        output = fill(NaN, D, predict_len)
        carried = Any[deepcopy(ctx.ensemble[i][2]) for i in 1:N]
        next_inputs = [copy(ctx.x0) for _ in 1:N]
        src = zeros(Int, N, nseg)
        bad_m = fill(false, N, nseg); donor_m = zeros(Int, N, nseg); anc_m = zeros(Int, N, nseg)
        active_m = fill(false, N, nseg); scores_m = fill(NaN, N, nseg)
        recorded = Dict{Symbol,Matrix{Float64}}(c => fill(NaN, N, nseg) for c in opts.record_criteria)
        ncorr = zeros(Int, N); lastc = zeros(Int, N); anc = collect(1:N); active = fill(true, N); Ks = zeros(Int, nseg)
    end

    cands = Vector{Matrix{Float64}}(undef, N)
    st_auto = Vector{Any}(undef, N)
    geom = fill(NaN, N, 2)
    own = (segs=Int[], bad=Vector{Int}[], cand=Array{Float64,3}[])     # 이번 chunk의 자기 출력 (교정 멤버만); 재개는 chunk 경계
    t_start = time()
    for s in s0:nseg
        r = seg_range(s, τ)
        truth_seg = view(ctx.truth, :, r)
        generate_candidates!(cands, st_auto, geom, ctx, carried, next_inputs, active, need_geom)
        fin = [active[i] && all(isfinite, cands[i]) for i in 1:N]
        pool = SegmentPool(cands, fin, geom, s, opts.random_seed)
        for c in opts.record_criteria
            recorded[c][:, s] .= criterion_scores(c, pool, ctx)
        end
        n_act = count(active)
        K = clamp(round(Int, n_act * opts.p), 1, max(n_act - 1, 1))
        if opts.correction_mode === :prune
            K = min(K, max(n_act - opts.min_active, 0))
        end
        Ks[s] = K
        bad = Int[]
        if opts.apply_to_all
            bad = findall(active)
        elseif opts.bad_selector !== nothing
            score_ctx = opts.bad_selector in LOOKAHEAD_SELECTORS ?
                merge(ctx, (selector_runtime=(hist=hist, src=src, st_auto=st_auto, p=opts.p),)) : ctx
            sc = criterion_scores(opts.bad_selector, pool, score_ctx)
            scores_m[:, s] .= sc
            bad = K > 0 ? select_bad(sc, fin, active, K) : [i for i in findall(active) if !fin[i]]
        end
        bad_m[bad, s] .= true
        badset = Set(bad)
        good = [i for i in findall(active) if !(i in badset)]
        for i in 1:N                                   # 모든 멤버: 자기 후보로 slot/state 진행
            hist[i][:, hcols(r)] .= cands[i]
            src[i, s] = i
            carried[i] = st_auto[i]
            next_inputs[i] = cands[i][:, end]
        end
        if correcting && !isempty(bad)
            donor = assign_donors(bad, opts.apply_to_all ? findall(active) : good, opts, s)
            src_old = src[:, 1:s]                      # 이번 segment 교정 전 lineage snapshot
            anc_old = copy(anc)
            corrected = sort(collect(keys(donor)))
            push!(own.segs, s); push!(own.bad, copy(corrected))
            push!(own.cand, isempty(corrected) ? Array{Float64,3}(undef, D, τ, 0) : cat((cands[j] for j in corrected)...; dims=3))
            for j in corrected
                d = donor[j]
                hist[j][:, r] .= cands[d]
                src[j, 1:(s - 1)] .= view(src_old, d, 1:(s - 1))
                src[j, s] = j
                anc[j] = anc_old[d]
                donor_m[j, s] = d
                ncorr[j] += 1
                lastc[j] = s
            end
            Threads.@threads for j in corrected
                ps, _, esn = ctx.ensemble[j]
                Y = assemble_history(hist, src, j, s, τ)
                carried[j] = correction_warmup(ctx, opts, j, Y)
                next_inputs[j] = Y[:, end]
            end
        end
        eff = [hist[i][:, hcols(r)] for i in 1:N]
        active_m[:, s] .= active
        rep = any(fin) ? nearest_to_consensus([active[i] ? eff[i] : fill(NaN, D, τ) for i in 1:N], τ) : nothing
        rep !== nothing && (output[:, r] .= eff[rep])
        record_segment_stats!(S, s, eff, truth_seg, rep, ctx, active)
        if opts.correction_mode === :prune
            for j in bad
                active[j] = false
            end
        end
        anc_m[:, s] .= anc

        if s % ck == 0
            c = s ÷ ck
            cols = correcting ? (((c - 1) * block_len + 1):(c * block_len)) : (1:block_len)
            blk = Array{Float64,3}(undef, D, block_len, N)
            for i in 1:N
                blk[:, :, i] .= view(hist[i], :, cols)
            end
            if correcting
                atomic_save(own_path(dir, c); segs=copy(own.segs), bad=copy(own.bad), cand=copy(own.cand))
                empty!(own.segs); empty!(own.bad); empty!(own.cand)
            end
            atomic_save(chunk_path(dir, c); block=blk)
            if s < nseg
                atomic_save(state_path; state=(seg=s, opts_fp=fp, stats=S, output=output, carried=carried,
                    next_inputs=next_inputs, src=src, bad=bad_m, donor=donor_m, ancestor=anc_m, active=active_m,
                    scores=scores_m, recorded=recorded, ncorr=ncorr, lastc=lastc, anc=anc, active_now=active, K=Ks))
            end
            log_line!(ctx.runlog, @sprintf("[%s] seg %d/%d  (bad=%d, active=%d, finite=%d, lineages=%d, %.1fs)",
                label, s, nseg, length(bad), count(active), count(fin), length(unique(anc[active])), time() - t_start))
            stop_after !== nothing && s == stop_after && s < nseg && return nothing
        end
        heartbeat!(ctx.runlog, "rollout_$(label)"; extra=Dict("segment" => s, "target" => nseg))
    end

    lt = NaN
    if correcting && !isempty(lineage_test_members)
        lt = lineage_check(ctx, opts, hist, src, lastc, carried, next_inputs, [i for i in lineage_test_members if active[i]])
        log_line!(ctx.runlog, "[$label] lineage self-test on $(length(lineage_test_members)) members: maxabsdiff=$lt")
    end
    result = (label=label, opts_fp=fp, stats=S, output=output, bad=bad_m, donor=donor_m, ancestor=anc_m,
        active=active_m, scores=scores_m, recorded=recorded, n_corrections=ncorr, last_corrected=lastc, K=Ks,
        src=src, lineage_test=lt, final_states=carried, final_inputs=next_inputs, elapsed_sec=time() - t_start)
    atomic_save(result_path; result=result)
    isfile(state_path) && rm(state_path)
    log_line!(ctx.runlog, @sprintf("[%s] complete (%.1fs)", label, time() - t_start))
    return result
end

"""
멤버 i의 현재 state를 lineage만으로 재구성: Y = assemble_history(i, n_seg). 마지막 교정 segment c(= last_corrected)가
있으면 state = (전체 이력 또는 최근 W) 재생(Y[:, 1:c·τ]) 후 Y[:, c·τ : end−1]을 teacher-forcing, 없으면 학습 직후
state에서 Y 전체 teacher-forcing. 이후 30 step 자율 출력을 실제 carried state의 것과 비교 → 최대 오차.
"""
function lineage_check(ctx, opts, hist, src, lastc, carried, next_inputs, members)
    τ = ctx.τ; nseg = ctx.n_segments
    worst = 0.0
    for i in members
        ps, _, esn = ctx.ensemble[i]
        Y = assemble_history(hist, src, i, nseg, τ)
        @assert Y[:, end] == next_inputs[i] "lineage next_input mismatch for member $i"
        c = lastc[i]
        st = if c == 0
            warmup_full_history(ps, ctx.init_states[i], esn, Y; x0=ctx.x0)
        else
            Yc = Y[:, 1:(c * τ)]
            s1 = correction_warmup(ctx, opts, i, Yc)
            c == nseg ? s1 : TestUtils.warmup_teacher_forcing(ps, s1, esn, Y[:, (c * τ):(size(Y, 2) - 1)])
        end
        a, _ = TestUtils.rollout_autonomous(ps, deepcopy(carried[i]), esn, 30, next_inputs[i])
        b, _ = TestUtils.rollout_autonomous(ps, st, esn, 30, next_inputs[i])
        worst = max(worst, maximum(abs.(a .- b)))
    end
    return worst
end

"""
    run_oracle(ctx, dir; ckpt_every=10) -> result   (상한선; truth 사용)

매 segment truth에 가장 가까운 후보(scaled sqerr 최소)로 전원을 re-warm (segment 단위 teacher-forcing;
StatementI :oracle_nmse와 동일). chunk 없음, output = 선택된 후보를 이은 궤적.
"""
function run_oracle(ctx, dir::AbstractString; ckpt_every::Int=10)
    mkpath(dir)
    rp = joinpath(dir, "result.jld2")
    isfile(rp) && return JLD2.load(rp, "result")
    N = ctx.n; τ = ctx.τ; nseg = ctx.n_segments; D = size(ctx.truth, 1)
    S = new_stats(N, D, nseg)
    output = fill(NaN, D, τ * nseg)
    carried = Any[deepcopy(ctx.ensemble[i][2]) for i in 1:N]
    next_inputs = [copy(ctx.x0) for _ in 1:N]
    cands = Vector{Matrix{Float64}}(undef, N); st_auto = Vector{Any}(undef, N); geom = fill(NaN, N, 2)
    t0 = time()
    for s in 1:nseg
        r = seg_range(s, τ)
        truth_seg = view(ctx.truth, :, r)
        generate_candidates!(cands, st_auto, geom, ctx, carried, next_inputs, fill(true, N), false)
        sc = [all(isfinite, cands[i]) ? scaled_sqerr_sum(cands[i], truth_seg, ctx.variance_scale) : Inf for i in 1:N]
        best = argmin(sc)
        B = cands[best]
        output[:, r] .= B
        Threads.@threads for i in 1:N
            ps, _, esn = ctx.ensemble[i]
            carried[i] = TestUtils.warmup_teacher_forcing(ps, carried[i], esn, hcat(next_inputs[i], view(B, :, 1:(τ - 1))))
            next_inputs[i] = B[:, end]
        end
        record_segment_stats!(S, s, cands, truth_seg, best, ctx)
        s % ckpt_every == 0 && log_line!(ctx.runlog, "[oracle_nmse] seg $s/$nseg")
    end
    result = (label="oracle_nmse", stats=S, output=output, elapsed_sec=time() - t0)
    atomic_save(rp; result=result)
    return result
end

"""
run 디렉터리의 chunk들로 멤버 `ids`(run 안의 index)의 유효 궤적 1:c_last·block을 꺼낸다. `own=true`면 교정된 칸을
own_XXX.jld2의 자기 candidate로 바꾼 "자기 출력" 궤적 (own 파일이 없는 chunk는 유효 기록 그대로 = 교정 없음).
"""
slim_path(dir::AbstractString) = joinpath(dir, "slim.jld2")

"""
Trajectories from slim.jld2 (Float64 copies): subset members in full; other members NaN except the tail. `own` applies the
stored own-record patches (complete for subset members and inside the tail).
"""
function load_slim_trajectories(dir::AbstractString, ids::AbstractVector{Int}; own::Bool=false, upto::Union{Nothing,Int}=nothing)
    d = JLD2.load(slim_path(dir))
    D, T = size(d["traj"], 1), d["T"]; τ = d["τ"]; ts = d["tail_start"]
    pos = Dict(j => a for (a, j) in enumerate(d["members"]))
    out = Matrix{Float64}[]
    for i in ids
        X = fill(NaN, D, T)
        haskey(pos, i) && (X .= view(d["traj"], :, :, pos[i]))
        X[:, ts:T] .= view(d["tail"], :, :, i)
        if own
            for (s, j, cand) in d["own_patches"]
                j == i || continue
                cols = ((s - 1) * τ + 1):(s * τ)
                (haskey(pos, i) || first(cols) >= ts) && (X[:, cols] .= cand)
            end
        end
        push!(out, upto === nothing ? X : X[:, 1:min(T, upto)])
    end
    return out
end

function load_member_trajectories(dir::AbstractString, ids::AbstractVector{Int}; upto_chunk::Union{Nothing,Int}=nothing,
                                  own::Bool=false)
    files = sort(filter(f -> occursin(r"^chunk_\d+\.jld2$", f), readdir(dir)))
    if isempty(files) && isfile(slim_path(dir))                 # chunks removed by src/analysis/slim.jl
        d_len = JLD2.load(slim_path(dir), "chunk_len")
        return load_slim_trajectories(dir, ids; own=own, upto=upto_chunk === nothing ? nothing : upto_chunk * d_len)
    end
    nc = upto_chunk === nothing ? length(files) : upto_chunk
    blocks = [JLD2.load(chunk_path(dir, c), "block") for c in 1:nc]
    if own
        for c in 1:nc
            isfile(own_path(dir, c)) || continue
            d = JLD2.load(own_path(dir, c))
            τ = size(first(d["cand"]), 2)
            for (k, s) in enumerate(d["segs"])
                cols = ((s - 1) * τ + 1 - (c - 1) * size(blocks[c], 2)):(s * τ - (c - 1) * size(blocks[c], 2))
                for (a, j) in enumerate(d["bad"][k])
                    blocks[c][:, cols, j] .= view(d["cand"][k], :, :, a)
                end
            end
        end
    end
    return [reduce(hcat, [b[:, :, i] for b in blocks]) for i in ids]
end

"chunk c의 블록(D × block_len × N, 유효 기록)에 own_XXX.jld2의 자기 candidate를 덮어 자기 출력 기록으로 (own 파일이 없으면 그대로)."
function apply_own_block!(blk::AbstractArray{<:Real,3}, dir::AbstractString, c::Int)
    isfile(own_path(dir, c)) || return blk
    d = JLD2.load(own_path(dir, c))
    bl = size(blk, 2)
    for (k, s) in enumerate(d["segs"])
        τ = size(d["cand"][k], 2)
        cols = ((s - 1) * τ + 1 - (c - 1) * bl):(s * τ - (c - 1) * bl)
        for (a, j) in enumerate(d["bad"][k])
            blk[:, cols, j] .= view(d["cand"][k], :, :, a)
        end
    end
    return blk
end

"chunk c 하나를 멤버별 행렬 Vector로 (D × block_len each)."
load_chunk_members(dir::AbstractString, c::Int) = (blk = JLD2.load(chunk_path(dir, c), "block");
    [blk[:, :, i] for i in axes(blk, 3)])

"""
    washout_length(ctx; members=1:min(20, ctx.n), perturb_len=500, run_len=3000, tol=1e-6) -> Vector{Int}

reservoir washout (training 데이터만 사용): 멤버마다 두 초기 state — 학습 직후 state A, 그리고 A에서 training 입력
앞부분 `perturb_len` step을 teacher-forcing한 다른 state B — 에 같은 training 입력열(`run_len` step)을 teacher-forcing
해서 1-step 출력 차이가 이후 계속 tol·(출력 규모) 아래로 떨어지는 첫 step을 잰다 (수렴 못 하면 run_len).
"""
function washout_length(ctx; members=1:min(20, ctx.n), perturb_len::Int=500, run_len::Int=3000, tol::Real=1e-6)
    U = ctx.train_input
    inp = U[:, (perturb_len + 1):min(size(U, 2), perturb_len + run_len)]
    scale = maximum(abs, ctx.train_target)
    out = zeros(Int, length(members))
    Threads.@threads for k in eachindex(members)
        ps, st, esn = ctx.ensemble[members[k]]
        stB = TestUtils.warmup_teacher_forcing(ps, deepcopy(ctx.init_states[members[k]]), esn, U[:, 1:perturb_len])
        ya, _ = ReservoirComputing.predict(esn, inp, ps, deepcopy(ctx.init_states[members[k]]))
        yb, _ = ReservoirComputing.predict(esn, inp, ps, stB)
        dif = vec(maximum(abs.(ya .- yb); dims=1))
        last_bad = findlast(>(tol * scale), dif)
        out[k] = last_bad === nothing ? 0 : last_bad
    end
    return out
end

# ------------------------------------------------------------
# 10. Self-tests
# ------------------------------------------------------------

function self_test_segmentwise_rollout(ctx; tol=1e-12)
    ps, st0, esn = ctx.ensemble[1]
    Ls = 50
    full, _ = TestUtils.rollout_autonomous(ps, deepcopy(st0), esn, 3Ls, ctx.x0)
    st = deepcopy(st0); x = copy(ctx.x0); parts = Matrix{Float64}[]
    for _ in 1:3
        o, st = TestUtils.rollout_autonomous(ps, st, esn, Ls, x)
        push!(parts, Matrix{Float64}(o))
        x = o[:, end]
    end
    d = maximum(abs.(reduce(hcat, parts) .- full))
    @assert d <= tol "self_test_segmentwise_rollout FAILED (maxabsdiff=$d)"
    return d
end

"자율 rollout 400 step 뒤의 state == 같은 출력의 teacher-forcing 재생 state (이후 50 step 출력으로 비교)."
function self_test_warmup_full_history(ctx; tol=1e-12)
    ps, st0, esn = ctx.ensemble[1]
    Y, st_auto = TestUtils.rollout_autonomous(ps, deepcopy(st0), esn, 400, ctx.x0)
    Y = Matrix{Float64}(Y)
    st_tf = warmup_full_history(ps, st0, esn, Y; x0=ctx.x0)
    a, _ = TestUtils.rollout_autonomous(ps, st_auto, esn, 50, Y[:, end])
    b, _ = TestUtils.rollout_autonomous(ps, st_tf, esn, 50, Y[:, end])
    d = maximum(abs.(a .- b))
    @assert d <= tol "self_test_warmup_full_history FAILED (maxabsdiff=$d)"
    return d
end

function self_test_rollout_with_states(ctx; tol=1e-10)
    ps, st0, esn = ctx.ensemble[1]
    a, sta = TestUtils.rollout_autonomous(ps, deepcopy(st0), esn, 200, ctx.x0)
    b, _, stb = _rollout_with_states(esn, 200, ps, deepcopy(st0); initialdata=ctx.x0)
    a2, _ = TestUtils.rollout_autonomous(ps, sta, esn, 50, a[:, end])
    b2, _ = TestUtils.rollout_autonomous(ps, stb, esn, 50, b[:, end])
    d = max(maximum(abs.(a .- b)), maximum(abs.(a2 .- b2)))
    @assert d <= tol "self_test_rollout_with_states FAILED (maxabsdiff=$d)"
    return d
end

