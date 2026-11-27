# ============================================================
# blind_selection_engine.jl
#
# Statement C가 도입한 "segment 단위 rollout → (ground-truth를 보거나/보지
# 않고) candidate 선택 → 선택된 candidate로 전체 ensemble re-warmup" 엔진과,
# 여러 Statement가 공유하는 selector 빌더들. Statement C뿐 아니라 Statement D
# (ensemble size sweep에서 같은 selector들을 비교)도 이 파일을 include한다 --
# CLAUDE.md 컨벤션상 StatementX.jl이 StatementY.jl을 직접 include하는 건
# 금지이므로, 두 Statement가 공유하는 로직은 여기로 옮겨 둔다.
#
# 여기 있는 것: 엔진 본체(`predict_ensemble_blind_rewarmup`)와, 최소
# 두 Statement 이상에서 쓰이는 selector 빌더(oracle/stability_filter/
# nearest_centroid/reservoir_geometry_*)와 그 보조 함수(`series_moments`).
# Statement C에서만 쓰는 selector(compression_to_train/wasserstein_to_train/
# spectral_stat_to_train/leverage_diversity)와 그 전용 헬퍼(스펙트럼 관련
# 함수 등)는 StatementC.jl에 그대로 남아 있다.
# ============================================================

using Statistics
using LinearAlgebra
using JLD2

# ------------------------------------------------------------
# Rollout engine
# ------------------------------------------------------------

"""
    predict_ensemble_blind_rewarmup(steps, ensemble; initialdata, rollout_len, selector, verbose=true)

Segment 단위로 모든 ensemble 멤버의 candidate rollout을 생성하고,
`selector(candidates, seg_range; next_input, carried_states, ensemble) -> (best_idx, score)`가
고른 candidate로 이어붙인 뒤, 그 candidate로 모든 멤버를 re-warmup하고 다음 segment로
진행한다. `next_input`(이 segment가 시작한 입력)과 `carried_states`(이 segment 시작 전
멤버별 reservoir state)는 candidate가 실제로 어떻게 만들어졌는지 재현해야 하는 selector
(예: `reservoir_geometry_*` selector들이 candidate를 `collectstates`로 재생할 때)를 위한 것 -- 이 정보가
필요 없는 selector는 `; kwargs...`로 무시하면 된다.

이 함수 자체는 truth 정보를 전혀 요구하지 않는다 — ground truth를 쓰는지 여부는
전적으로 `selector` 구현에 달려 있다 (비교용 oracle selector도 이 엔진으로 실행한다).

`on_progress(seg, n_segments)`는 매 segment가 끝날 때마다 호출된다 (로깅/heartbeat 용도).

`checkpoint_path`가 주어지면 매 `checkpoint_interval` segment마다 진행 상황
(`output`/`selected_indices`/`selected_scores`/`segment_ranges`/`carried_states`/
`next_input`)을 tmp-then-rename으로 저장하고, 함수 시작 시 그 파일이 있으면
다음 segment부터 이어서 돈다 -- 이 함수를 감싸는 `load_or_compute`는 완주해야만
결과를 캐싱하므로, 체크포인트가 없으면 크래시 시점까지의 segment들이 전부
날아간다. 완주하면 체크포인트 파일은 지운다.
"""
function predict_ensemble_blind_rewarmup(
    steps::Int,
    ensemble::AbstractVector;
    initialdata::AbstractVector,
    rollout_len::Int,
    selector,  # plain (candidates, seg_range; ...) -> (best_idx, score) 함수 또는 FusedGeometrySelector
    verbose::Bool=true,
    on_progress::Function=(seg, n) -> nothing,
    checkpoint_path::Union{Nothing,AbstractString}=nothing,
    checkpoint_interval::Int=1,
)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    rollout_len > 0 || throw(ArgumentError("rollout_len must be positive"))
    isempty(ensemble) && throw(ArgumentError("ensemble must not be empty"))

    out_size = length(initialdata)
    n_members = length(ensemble)
    n_segments = cld(steps, rollout_len)

    output = Matrix{Float64}(undef, out_size, steps)

    selected_indices = Vector{Int}(undef, n_segments)
    selected_scores = Vector{Float64}(undef, n_segments)
    segment_ranges = Vector{UnitRange{Int}}(undef, n_segments)

    carried_states = [deepcopy(ensemble[i][2]) for i in 1:n_members]
    next_input = copy(initialdata)
    seg_start_idx = 1

    if checkpoint_path !== nothing && isfile(checkpoint_path)
        ckpt = JLD2.load(checkpoint_path)
        (ckpt["steps"] == steps && ckpt["rollout_len"] == rollout_len && ckpt["n_members"] == n_members) || throw(ArgumentError(
            "blind-rewarmup checkpoint at \"$checkpoint_path\" doesn't match this call " *
            "(steps/rollout_len/n_members differ) -- delete it or use a fresh path."
        ))
        output = ckpt["output"]
        selected_indices = ckpt["selected_indices"]
        selected_scores = ckpt["selected_scores"]
        segment_ranges = ckpt["segment_ranges"]
        carried_states = ckpt["carried_states"]
        next_input = ckpt["next_input"]
        seg_start_idx = ckpt["seg_done"] + 1
        verbose && println("predict_ensemble_blind_rewarmup: resuming from checkpoint (segment $(ckpt["seg_done"])/$n_segments)")
    end

    progress = Progress(n_segments; desc="Blind ensemble rollout L=$rollout_len")

    for seg in seg_start_idx:n_segments
        seg_start = (seg - 1) * rollout_len + 1
        seg_end = min(seg_start + rollout_len - 1, steps)
        current_len = seg_end - seg_start + 1
        seg_range = seg_start:seg_end

        # 멤버마다 완전히 독립된 rollout이라 스레드 병렬화 가능
        # (candidates[i]/carried_states[i]만 읽고 씀, 겹치는 슬롯 없음).
        candidates = Vector{Matrix{Float64}}(undef, n_members)
        if selector isa FusedGeometrySelector
            # geometry selector: candidate 생성과 동시에 hidden-state를 얻어
            # (`_rollout_with_states`) 그 자리에서 바로 점수까지 매긴다 -- 예전처럼
            # candidate를 다 만든 뒤 selector가 같은 recurrence를 `collectstates`로
            # 한 번 더 돌 필요가 없다. state 행렬은 멤버당 이 iteration 안에서만
            # 살아있다가 버려지므로(스레드 개수만큼만 동시 존재) 메모리 사용량은
            # 이 최적화 이전과 동일하다 (모든 멤버 state를 동시에 들고 있지 않음).
            scores = Vector{Float64}(undef, n_members)
            Threads.@threads for i in 1:n_members
                ps, _, esn = ensemble[i]
                cand_i, states_i, _ = _rollout_with_states(
                    esn, current_len, ps, carried_states[i]; initialdata=next_input,
                )
                candidates[i] = Matrix{Float64}(cand_i)
                scores[i] = selector.score_member(i, states_i)
            end
            best_idx = argmin(scores)
            score = scores[best_idx]
        else
            Threads.@threads for i in 1:n_members
                ps, _, esn = ensemble[i]
                candidate_i, _ = ReservoirComputing.predict(
                    esn,
                    current_len,
                    ps,
                    carried_states[i];
                    initialdata=next_input,
                )
                candidates[i] = Matrix{Float64}(candidate_i)
            end

            best_idx, score = selector(
                candidates, seg_range;
                next_input=next_input, carried_states=carried_states, ensemble=ensemble,
            )
        end
        best_seg = candidates[best_idx]
        output[:, seg_range] .= best_seg

        selected_indices[seg] = best_idx
        selected_scores[seg] = score
        segment_ranges[seg] = seg_range

        Threads.@threads for i in 1:n_members
            ps, _, esn = ensemble[i]
            carried_states[i] = TestUtils.warmup_teacher_forcing(
                ps,
                carried_states[i],
                esn,
                best_seg,
            )
        end

        next_input = copy(view(best_seg, :, size(best_seg, 2)))

        verbose && println(
            "L=$rollout_len, segment $seg ($(first(seg_range)):$(last(seg_range))): " *
            "best=$best_idx, score=$score"
        )

        next!(progress)
        on_progress(seg, n_segments)

        if checkpoint_path !== nothing && seg % checkpoint_interval == 0
            tmp_path = checkpoint_path * ".tmp"
            jldsave(tmp_path;
                seg_done=seg, output=output, selected_indices=selected_indices,
                selected_scores=selected_scores, segment_ranges=segment_ranges,
                carried_states=carried_states, next_input=next_input,
                steps=steps, rollout_len=rollout_len, n_members=n_members,
            )
            mv(tmp_path, checkpoint_path; force=true)
        end
    end

    if checkpoint_path !== nothing && isfile(checkpoint_path)
        rm(checkpoint_path)
    end

    metadata = (
        rollout_len=rollout_len,
        selected_indices=selected_indices,
        selected_scores=selected_scores,
        segment_ranges=segment_ranges,
    )

    return output, metadata
end

# ------------------------------------------------------------
# Fused rollout+state helper
#
# reservoir_geometry_* selector들은 candidate의 hidden-state 궤적(X_test)이
# 필요한데, `ReservoirComputing.predict`(candidate 생성에 쓰는 autoregressive
# rollout)는 readout 출력만 반환하고 중간 reservoir feature는 버린다. 예전에는
# 그래서 candidate를 다 만든 뒤 `collectstates`로 candidate 자신의 출력을 다시
# teacher-forced 입력으로 넣어 같은 recurrence를 한 번 더 돌려 state를 얻었다
# (`replay_input = hcat(next_input, cand[:, 1:end-1])`) -- 즉 세그먼트마다
# 멤버당 O(res_size^2 * rollout_len) recurrence를 두 번 반복하고 있었다.
#
# `_reservoir_feature`/`_rollout_with_states`는 `ReservoirComputing.apply`
# (공개 API)와 `ESN`/`ReservoirComputer` docstring에 문서화된 필드
# (`esn.reservoir`/`esn.states_modifiers`/`esn.readout`)만으로 그 내부
# recurrence(`_partial_apply` + readout, 매 step 자기 출력을 다음 입력으로
# 되먹임)를 재구현해, 같은 autoregressive rollout 한 번으로 출력과 state를
# 동시에 기록한다. `predict` 한 번 + `collectstates` replay 한 번과 결과가
# 수치적으로 동일함(state_modifiers 0개/2개 케이스 모두 ~1e-16 오차, 즉
# floating-point 재정렬 수준)을 별도 스크립트로 검증했다.
# ------------------------------------------------------------

# ReservoirComputing.jl는 버전에 따라 modifier 필드 이름이 다르다: 0.12.24 등은
# `states_modifiers`(복수), 0.12.32부터는 다시 `state_modifiers`(단수). ESN struct는
# 옛 이름도 getproperty alias로 받아주지만 `ps`/`st` NamedTuple은 그렇지 않으므로,
# 설치된 버전이 실제로 쓰는 키를 그대로 읽고 같은 키로 돌려준다.
_mods_key(nt) = hasproperty(nt, :states_modifiers) ? :states_modifiers : :state_modifiers

function _reservoir_feature(esn, inp, ps, st)
    z, st_res = ReservoirComputing.apply(esn.reservoir, inp, ps.reservoir, st.reservoir)
    key = _mods_key(ps)
    mods = getproperty(esn, key)
    ps_mods = getproperty(ps, key)
    st_mods = getproperty(st, key)
    new_st_mods = Vector{Any}(undef, length(mods))
    for k in eachindex(mods)
        z, sk = ReservoirComputing.apply(mods[k], z, ps_mods[k], st_mods[k])
        new_st_mods[k] = sk
    end
    return z, NamedTuple{(:reservoir, key)}((st_res, Tuple(new_st_mods)))
end

"""
    _rollout_with_states(esn, steps, ps, st; initialdata) -> (output, states, new_st)

`ReservoirComputing.predict`와 동일한 autoregressive rollout(매 step 자기 readout
출력을 다음 입력으로 되먹임)을 수행하면서, 동시에 매 step의 reservoir feature도
`states`(res_size × steps)에 기록한다.
"""
function _rollout_with_states(esn, steps::Int, ps, st; initialdata::AbstractVector)
    next_input = initialdata

    z1, st1 = _reservoir_feature(esn, next_input, ps, st)
    y1, st_ro1 = ReservoirComputing.apply(esn.readout, z1, ps.readout, st.readout)
    cur_st = merge(st1, (readout=st_ro1,))

    states = zeros(eltype(z1), length(z1), steps)
    output = zeros(eltype(y1), length(y1), steps)
    states[:, 1] = z1
    output[:, 1] = y1
    next_input = y1

    for step in 2:steps
        z, stz = _reservoir_feature(esn, next_input, ps, cur_st)
        y, st_ro = ReservoirComputing.apply(esn.readout, z, ps.readout, cur_st.readout)
        cur_st = merge(stz, (readout=st_ro,))
        states[:, step] = z
        output[:, step] = y
        next_input = y
    end

    return output, states, cur_st
end

"""
    FusedGeometrySelector(score_member)

`make_reservoir_geometry_metric_selector`가 반환하는 selector의 실제 타입.
`predict_ensemble_blind_rewarmup`이 `selector isa FusedGeometrySelector`를 보고
candidate 생성 루프 안에서 바로 `_rollout_with_states` + `score_member(i,
states_i)`를 호출해 점수까지 매긴다 (일반 selector처럼 candidate를 다 만든 뒤
따로 호출되지 않음) -- `score_member`는 `i`번째 멤버의 state 궤적만 받아 점수
하나(작을수록 좋음)를 반환하면 된다.
"""
struct FusedGeometrySelector{F}
    score_member::F
end

# ------------------------------------------------------------
# Shared statistics helper (stability_filter/spectral_stat_to_train 둘 다 사용)
# ------------------------------------------------------------

function _skewness(x::AbstractVector)
    σ = std(x; corrected=false)
    σ == 0 && return 0.0
    μ = mean(x)
    return mean(((x .- μ) ./ σ) .^ 3)
end

function _kurtosis(x::AbstractVector)
    σ = std(x; corrected=false)
    σ == 0 && return 0.0
    μ = mean(x)
    return mean(((x .- μ) ./ σ) .^ 4) - 3.0
end

"""
Per-dimension mean/variance/skewness/excess-kurtosis of a (state × time) series.
"""
function series_moments(data::AbstractMatrix)
    d = size(data, 1)
    return (
        mean=vec(mean(data; dims=2)),
        var=vec(var(data; dims=2, corrected=false)),
        skew=[_skewness(view(data, i, :)) for i in 1:d],
        kurt=[_kurtosis(view(data, i, :)) for i in 1:d],
    )
end

# ------------------------------------------------------------
# Selector builders
# ------------------------------------------------------------

function make_oracle_selector(teacher_data::AbstractMatrix)
    variance_scale = global_truth_variance(teacher_data)
    return function (candidates, seg_range; kwargs...)
        truth_seg = view(teacher_data, :, seg_range)
        best_idx = 1
        best_dist = Inf
        for (i, cand) in enumerate(candidates)
            dist = mean(pointwise_nmse(truth_seg, cand; variance_scale=variance_scale))
            dist = isfinite(dist) ? dist : Inf
            if dist < best_dist
                best_dist = dist
                best_idx = i
            end
        end
        return best_idx, best_dist
    end
end

function make_nearest_centroid_selector(; embedding_config, feature_mode::Symbol=:mean, kmeans_k::Int=5)
    return function (candidates, seg_range; kwargs...)
        X = compute_candidate_features(candidates; embedding_config=embedding_config, feature_mode=feature_mode)
        best_idx, best_dist, _ = select_nearest_centroid(X, kmeans_k)
        return best_idx, best_dist
    end
end

function make_stability_filter_selector(
    train_moments,
    train_extent;
    stability_factor::Float64=4.0,
)
    margin = stability_factor .* sqrt.(train_moments.var)
    lo = train_extent.lo .- margin
    hi = train_extent.hi .+ margin

    return function (candidates, seg_range; kwargs...)
        survivors = Int[]
        for (i, cand) in enumerate(candidates)
            ok = all(isfinite, cand) &&
                 all(vec(minimum(cand; dims=2)) .>= lo) &&
                 all(vec(maximum(cand; dims=2)) .<= hi)
            ok && push!(survivors, i)
        end
        pool = isempty(survivors) ? collect(eachindex(candidates)) : survivors

        best_idx = pool[1]
        best_score = Inf
        for i in pool
            cand = candidates[i]
            score = if all(isfinite, cand)
                cand_var = vec(var(cand; dims=2, corrected=false))
                mean(abs.(log.(cand_var .+ eps()) .- log.(train_moments.var .+ eps())))
            else
                Inf
            end
            if score < best_score
                best_score = score
                best_idx = i
            end
        end
        return best_idx, best_score
    end
end

"""
`geometry_profiles[i]`(멤버 `i`의 `training_geometry_profile`, 멤버당 한 번만 계산해
캐싱해 둔 값)와, 이 segment에서 멤버 `i`가 실제로 만든 candidate의 hidden-state
궤적을 비교한다. `predict_ensemble_blind_rewarmup`이 이 selector가
`FusedGeometrySelector`임을 보고 candidate 생성 자체를 `_rollout_with_states`로
수행하므로(state와 출력을 한 rollout에서 동시에 얻음), 여기서는 이미 얻어진
`states_i`를 받아 점수만 계산한다 -- 예전처럼 candidate를 다 만든 뒤
`collectstates`로 같은 recurrence를 다시 돌 필요가 없다.

세 독립 metric(leverage_distance / subspace_residual / effective_dimension_mismatch, 모두
작을수록 좋음) 중 `metric` 하나만 골라 그 값이 최소인 candidate를 선택한다 --
`combined_geometry_score`처럼 세 지표를 robust z-score 후 합산하지 않고, 각 metric을
독립된 selector로 취급한다. `score_against_profile`(세 metric을 항상 다 계산)이 아니라
`leverage_distance_score`/`subspace_residual_score`/`effective_dimension_mismatch_score`
중 `metric`에 해당하는 것 하나만 호출한다 -- 특히 `effective_dimension_mismatch`만 SVD를
필요로 하므로(계산량 대부분이 여기 있다), `:leverage_distance`/`:subspace_residual`
selector는 이 SVD를 아예 계산하지 않는다.
"""
function make_reservoir_geometry_metric_selector(
    metric::Symbol,
    ensemble::AbstractVector,
    geometry_profiles::AbstractVector,
)
    score_fn = if metric === :leverage_distance
        leverage_distance_score
    elseif metric === :subspace_residual
        subspace_residual_score
    elseif metric === :effective_dimension_mismatch
        (profile, Z_test) -> first(effective_dimension_mismatch_score(profile, Z_test))
    else
        throw(ArgumentError("unknown reservoir-geometry metric: $metric"))
    end

    score_member = function (i, states_i)
        X_test = Matrix{Float64}(states_i)
        Z_test = profile_zscore(geometry_profiles[i], X_test)
        return score_fn(geometry_profiles[i], Z_test)
    end

    return FusedGeometrySelector(score_member)
end
