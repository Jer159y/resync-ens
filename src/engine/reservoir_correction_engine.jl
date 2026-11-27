# ============================================================
# reservoir_correction_engine.jl
#
# StatementE_correct.jl 의 "방식 C" (나쁜 reservoir 식별 → 좋은 것으로 교정)
# 로직을, 후속 분석 스크립트(StatementE_dropout_analysis.jl /
# StatementE_fair_compare.jl)가 "재구현 없이" 그대로 import 할 수 있도록 뽑아 둔
# 공유 파일. StatementE_correct.jl 이 이 파일을 include 하고 인라인 정의 대신
# 여기 함수를 쓴다.
#
# 여기 있는 것:
#   corr_fe1               — FE-1 (with_zero_crossing=false), local_var_window=10
#   identify_bad           — candidates → 나쁜 멤버 index 집합 (K개 + 발산 멤버)
#   correct_bad_reservoirs — 나쁜 멤버 → 교정용 trajectory (Dict)
#   segment_health         — segment pool 통계 (diversity + fidelity_rate)
#   nearest_to_consensus   — pool 중 consensus 최근접 원소 index
#
# 의존: src/prelude.jl (extract_stat_features via FeatureExtractors),
#       Statistics / LinearAlgebra / Random.
# ============================================================

using Statistics
using LinearAlgebra
using Random

const CORR_STAT_LOCAL_VAR_WINDOW = 10

"FE-1 shape 통계 (unbounded-scale only). StatementE_correct 의 `fe1` 와 동일 정의."
corr_fe1(t::AbstractMatrix) =
    extract_stat_features(t; local_var_window=CORR_STAT_LOCAL_VAR_WINDOW, with_zero_crossing=false)

"""
    identify_bad(candidates, K; criterion, tf) -> Set{Int}

candidates(Vector of state×τ) → 나쁜 멤버 index 집합 (상위 K개 + 발산(비유한) 멤버 전부).
`tf` = training-window FE-1 reference (`mean`/`scale` 필드 사용, `train_fe1(L)` 결과).

criterion:
  :state_divergence — ‖Tᵢ − T̄‖ 를 segment 전체에서 평균 (T̄ = finite candidate 평균)
  :tail_divergence  — 위와 같되 segment 후반 τ/5 구간만
  :score_based      — ‖(FE1(Tᵢ) − tf.mean) / tf.scale‖
"""
function identify_bad(candidates::AbstractVector, K::Int; criterion::Symbol, tf)
    n = length(candidates)
    τ = size(candidates[1], 2)
    finite = [all(isfinite, c) for c in candidates]
    Tmean = let good = [candidates[i] for i in 1:n if finite[i]]
        isempty(good) ? zeros(size(candidates[1])) : mean(good)
    end
    score = fill(Inf, n)
    if criterion === :state_divergence
        for i in 1:n
            finite[i] && (score[i] = mean(norm(view(candidates[i], :, t) .- view(Tmean, :, t)) for t in 1:τ))
        end
    elseif criterion === :tail_divergence
        w = max(1, τ ÷ 5)
        rng = (τ - w + 1):τ
        for i in 1:n
            finite[i] && (score[i] = mean(norm(view(candidates[i], :, t) .- view(Tmean, :, t)) for t in rng))
        end
    elseif criterion === :score_based
        for i in 1:n
            finite[i] && (score[i] = norm((corr_fe1(candidates[i]) .- tf.mean) ./ tf.scale))
        end
    else
        throw(ArgumentError("unknown criterion $criterion"))
    end
    order = sortperm(score; rev=true)
    bad = Set(order[1:min(K, n)])
    for i in 1:n
        finite[i] || push!(bad, i)
    end
    return bad
end

"""
    correct_bad_reservoirs(candidates, bad, good, method; rng, tf) -> Dict{Int, Matrix}

각 나쁜 멤버 → 교정에 쓸 trajectory (그걸로 re-warmup + next_input).

method:
  :random_each   — 나쁜 멤버마다 독립적으로 랜덤한 "좋은" 멤버 궤적 (diversity 유지)
  :distinct_each — good 셔플 후 round-robin (가능하면 중복 없이)
  :random_good   — 좋은 것 중 랜덤 1개를 모든 나쁜 멤버에 (diversity 죽음, 비교용)
  :good_mean     — 좋은 것들의 평균 궤적을 모든 나쁜 멤버에 (diversity 죽음, 비교용)
  :nearest_good  — feature space 에서 각 나쁜 멤버에 가장 가까운 좋은 것
"""
function correct_bad_reservoirs(candidates::AbstractVector, bad::Set{Int}, good::Vector{Int}, method::Symbol;
                                rng::AbstractRNG=Random.default_rng(), tf)
    isempty(good) && return Dict(i => candidates[i] for i in bad)
    out = Dict{Int,Matrix{Float64}}()
    if method === :random_each
        for i in bad
            out[i] = candidates[good[rand(rng, 1:length(good))]]
        end
    elseif method === :distinct_each
        order = shuffle(rng, good)
        for (k, i) in enumerate(sort(collect(bad)))
            out[i] = candidates[order[mod1(k, length(order))]]
        end
    elseif method === :random_good
        g = candidates[good[rand(rng, 1:length(good))]]
        for i in bad
            out[i] = g
        end
    elseif method === :good_mean
        g = mean(candidates[j] for j in good)
        for i in bad
            out[i] = g
        end
    elseif method === :nearest_good
        gf = Dict(j => corr_fe1(candidates[j]) for j in good)
        for i in bad
            fi = all(isfinite, candidates[i]) ? corr_fe1(candidates[i]) : fill(Inf, length(tf.mean))
            j = argmin([norm((gf[j] .- fi) ./ tf.scale) for j in good])
            out[i] = candidates[good[j]]
        end
    else
        throw(ArgumentError("unknown correction method $method"))
    end
    return out
end

"segment candidate pool 통계: diversity(finite 후보 평균 pairwise 거리) + fidelity_rate. `tf` = train_fe1(L)."
function segment_health(candidates::AbstractVector, tf)
    n = length(candidates)
    fin = [i for i in 1:n if all(isfinite, candidates[i])]
    V = [vec(candidates[i]) for i in fin]
    diversity = if length(V) >= 2
        acc = 0.0; cnt = 0
        for a in 1:length(V), b in (a + 1):length(V)
            acc += norm(V[a] - V[b]); cnt += 1
        end
        acc / cnt
    else
        NaN
    end
    iqr = tf.q3 .- tf.q1
    lo = tf.q1 .- 1.5 .* iqr
    hi = tf.q3 .+ 1.5 .* iqr
    fidelity = 0
    for i in fin
        f = corr_fe1(candidates[i])
        mean(lo .<= f .<= hi) >= 0.7 && (fidelity += 1)
    end
    return (diversity=diversity, fidelity_rate=fidelity / n)
end

"consensus(= finite candidate 평균)에 segment 전체에서 가장 가까운 pool 원소의 index."
function nearest_to_consensus(pool::AbstractVector, seglen::Int)
    fin = [all(isfinite, p) for p in pool]
    cen = let g = [pool[i] for i in eachindex(pool) if fin[i]]
        isempty(g) ? mean(pool) : mean(g)
    end
    scores = [fin[i] ? mean(norm(view(pool[i], :, t) .- view(cen, :, t)) for t in 1:seglen) : Inf
              for i in eachindex(pool)]
    return argmin(scores)
end
