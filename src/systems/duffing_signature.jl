# ============================================================
# duffing_signature.jl
#
# Duffing(ẍ + δẋ − αx + βx³ = γ cos ωt, data/generators/duffing_oscillator.jl의 Holmes 설정 α=β=1, δ=0.3,
# γ=0.4, ω=1.2, dt=0.1)에만 있는 성질로 "궤적이 Duffing다운가"를 보는 질적 보조 지표. 일반성은 없다
# (다른 시스템은 자기 signature 파일을 따로 둔다).
#
#   S1 well 체류 시간 (dwell time)  x1 부호 전환을 hysteresis(±θ)로 판정한 한 well 연속 체류 길이 분포.
#                                   카오틱 Duffing은 넓은(대략 지수) 꼬리; well-lock은 체류가 끝나지 않고,
#                                   주기 궤도는 한두 값에 몰린다. 멤버 지표 = log 체류 시간 분포의 W1(vs truth),
#                                   전환 빈도(1000 step당).
#   S2 에너지 H = x2²/2 − αx1²/2 + βx1⁴/4   well 바닥 H = −α²/4β, saddle(장벽) H = 0. 분포와 장벽 위 체류 비율.
#                                   멤버 지표 = W1(H 분포, truth H 분포).
#   S3 stroboscopic Poincaré section  외력 위상 ωt ≡ φ0 (mod 2π)에서 찍은 (x1, x2). 카오틱 Duffing은 프랙탈
#                                   (Holmes/Ueda) 구조; fixed point → 한 점, 주기 궤도 → 몇 개 점, 발산 → 밖.
#                                   멤버 지표 = 격자 히스토그램 JS divergence (vs 긴 기준 궤적의 section).
# 예측의 k번째 열은 truth의 k번째 열과 같은 물리 시각이므로 외력 위상도 truth 시각으로 정한다.
# ensemble_diversity_metrics.jl(grid_prob, js_divergence)을 먼저 include해야 한다.
# ============================================================

const DUFFING_PARAMS = (α=1.0, β=1.0, δ=0.3, γ=0.4, ω=1.2, dt=0.1)
const DUFFING_DWELL_THETA = 0.3            # hysteresis: x1 > θ → 오른쪽 well, x1 < −θ → 왼쪽 well

duffing_energy(X::AbstractMatrix; α=DUFFING_PARAMS.α, β=DUFFING_PARAMS.β) =
    [X[2, t]^2 / 2 - α * X[1, t]^2 / 2 + β * X[1, t]^4 / 4 for t in axes(X, 2)]

"""
    dwell_times(x1; θ) -> Vector{Int}

hysteresis 부호열(오른쪽 x1>θ, 왼쪽 x1<−θ, 그 사이는 직전 well 유지)에서 완결된 체류 길이(step)만.
첫 체류와 마지막(잘린) 체류는 뺀다.
"""
function dwell_times(x1::AbstractVector; θ::Real=DUFFING_DWELL_THETA)
    state = 0; start = 0; first_done = false
    out = Int[]
    for t in eachindex(x1)
        v = x1[t]
        isfinite(v) || return out
        new = v > θ ? 1 : v < -θ ? -1 : state
        if new != state
            state != 0 && (first_done ? push!(out, t - start) : (first_done = true))
            state = new; start = t
        end
    end
    return out
end

"x1 전환 횟수(hysteresis) / 1000 step."
function switch_rate(x1::AbstractVector; θ::Real=DUFFING_DWELL_THETA)
    state = 0; n = 0
    for v in x1
        isfinite(v) || return NaN
        new = v > θ ? 1 : v < -θ ? -1 : state
        (new != state && state != 0) && (n += 1)
        state = new
    end
    return 1000n / length(x1)
end

"외력 위상이 φ0에서 반 step 이내인 truth 열 번호 (time_of: 열 → 물리 시각)."
function section_columns(T::Int, time_of::Function; φ0::Real=0.0, ω=DUFFING_PARAMS.ω, dt=DUFFING_PARAMS.dt)
    hw = ω * dt / 2
    return [t for t in 1:T if abs(mod(ω * time_of(t) - φ0 + π, 2π) - π) <= hw]
end

"1D 표본 두 개의 W1 (분위 함수 L1; 길이가 다르면 공통 분위 격자 200개)."
function w1_samples(a::AbstractVector, b::AbstractVector; nq::Int=200)
    (isempty(a) || isempty(b)) && return NaN
    p = range(0.5 / nq, 1 - 0.5 / nq; length=nq)
    return mean(abs.(quantile(a, p) .- quantile(b, p)))
end

"""
    duffing_member_signature(X, sec_cols, truth_ref) -> NamedTuple

X = 멤버 궤적(2 × T). truth_ref = (log_dwell, H, sec_prob, lo, hi, nbins) — 기준 분포.
비유한 궤적이면 모든 값 NaN.
"""
function duffing_member_signature(X::AbstractMatrix, sec_cols::AbstractVector{Int}, truth_ref)
    all(isfinite, X) || return (switch_rate=NaN, dwell_w1=NaN, energy_w1=NaN, section_js=NaN, above_barrier=NaN,
        n_dwell=0)
    x1 = view(X, 1, :)
    dw = dwell_times(x1)
    H = duffing_energy(X)
    sp = grid_prob(X[:, sec_cols], truth_ref.lo, truth_ref.hi, truth_ref.nbins)
    return (switch_rate=switch_rate(x1), dwell_w1=length(dw) >= 2 ? w1_samples(log10.(dw), truth_ref.log_dwell) : NaN,
        energy_w1=w1_samples(H, truth_ref.H), section_js=js_divergence(sp, truth_ref.sec_prob),
        above_barrier=mean(>(0), H), n_dwell=length(dw))
end

function self_test_duffing_signature()
    # 오른쪽 100 step, 왼쪽 50 step 반복 → 완결 체류 = 100, 50, …
    x = vcat(repeat(vcat(fill(1.0, 100), fill(-1.0, 50)), 5))
    d = dwell_times(x)
    @assert d == [50, 100, 50, 100, 50, 100, 50, 100] "dwell_times $d"
    @assert switch_rate(x) ≈ 1000 * 9 / length(x)
    @assert duffing_energy([1.0 0.0; 0.0 0.0])[1] ≈ -0.25
    @assert w1_samples(collect(1.0:100), collect(1.0:100)) < 1e-12
    println("self_test_duffing_signature: OK")
end

"""
    duffing_lyapunov(; T=20000.0, h=0.01, renorm=0.1, transient=500.0) -> λ_max (1/시간)

실제 Duffing 방정식(외력 포함, RK4 step h)에 Benettin 두-궤적 재정규화 (renorm 시간마다). 데이터 간격 dt=0.1이므로
step당 값은 λ·0.1.
"""
function duffing_lyapunov(; T::Real=20000.0, h::Real=0.01, renorm::Real=0.1, transient::Real=500.0, d0::Real=1e-8)
    p = DUFFING_PARAMS
    f(u, t) = [u[2], -p.δ * u[2] + p.α * u[1] - p.β * u[1]^3 + p.γ * cos(p.ω * t)]
    rk4(u, t) = (k1 = f(u, t); k2 = f(u .+ h / 2 .* k1, t + h / 2); k3 = f(u .+ h / 2 .* k2, t + h / 2);
                 k4 = f(u .+ h .* k3, t + h); u .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4))
    u = [1.0, 0.0]; t = 0.0
    for _ in 1:round(Int, transient / h)
        u = rk4(u, t); t += h
    end
    v = u .+ [d0, 0.0]
    m = round(Int, renorm / h); acc = 0.0; n = 0
    for _ in 1:round(Int, T / renorm)
        for _ in 1:m
            u = rk4(u, t); v = rk4(v, t); t += h
        end
        d = norm(v .- u); acc += log(d / d0); n += 1
        v = u .+ (d0 / d) .* (v .- u)
    end
    return acc / (n * renorm)
end
