using ProgressMeter: @showprogress
using Random
using JLD2
using Logging: with_logger, NullLogger
using ReservoirComputing: ESN, rand_sparse, scaled_rand, zeros32
using SparseArrays                      # rand_sparse(...; return_sparse=true) needs it loaded

"""
    _jldsave_quiet(path; kwargs...)

`esn`이 포함된 checkpoint를 저장할 때 쓰는 `jldsave` 래퍼. `_train_one_member`/
`migrate_legacy_member`가 만드는 `init_state = (rng, dims...) -> randn(...)` 클로저는
캡처하는 값이 없는(stateless) 순수 익명 함수라서, JLD2가 이를 저장할 때마다
"JLD2 only stores functions by name" 경고를 낸다 -- 캡처된 상태가 없으니 이름으로만
저장해도 정보 손실은 없다(로드 시 같은 소스 위치의 익명 함수가 재구성됨). 멤버
저장이 `n_ensemble`번(많게는 15000번) 반복되므로 콘솔만 심하게 어지럽혀 저장
시에만 로거를 끈다. `jldsave` 자체가 던지는 실제 에러(디스크 I/O 실패 등)는 로그가
아니라 예외이므로 이 억제와 무관하게 그대로 전파된다.
"""
function _jldsave_quiet(path; kwargs...)
    with_logger(NullLogger()) do
        jldsave(path; kwargs...)
    end
end

"""
    train_ensemble(
        input_data::Matrix,
        target_data::Matrix,
        n_ensemble::Int;
        base_config::NamedTuple,
        vary_seed::Bool=true,
        seed_start::Int=1234
    ) -> Vector{Tuple}

여러 ESN 모델을 학습하여 ensemble 생성.
각 모델은 동일한 하이퍼파라미터를 가지며 random seed만 변경됨.

# Arguments
- `input_data`: 학습용 입력 데이터 (in_size × train_len)
- `target_data`: 학습용 타겟 데이터 (out_size × train_len)
- `n_ensemble`: 앙상블을 구성할 ESN 모델 개수
- `base_config`: 기본 ESN 설정 (dict 또는 NamedTuple)
  - Required: in_size, res_size, out_size, training_method, washout, steps
  - Optional: radius, sparsity, scaling, leak_coefficient, etc.
- `vary_seed`: true이면 seed 변경, false이면 동일 seed 사용
- `seed_start`: 시작 seed 값

# Returns
- Vector of (ps, st) tuples (파라미터와 상태)
"""

function _train_one_member(input_data, target_data, base_config, seed::Int; sparse::Bool=false)
    rng = MersenneTwister(seed)

    # sparse = true stores the same reservoir matrix as SparseMatrixCSC (identical entries; readout and rollouts
    # agree to ~1e-12). Much faster for large reservoirs, especially with many members on threads.
    init_reservoir = rand_sparse(; radius=base_config.radius, sparsity=base_config.sparsity, return_sparse=sparse)
    init_input = scaled_rand(; scaling=base_config.scaling)
    init_state = (rng, dims...) -> randn(rng, Float32, dims...)

    esn = ESN(base_config.in_size,
            base_config.res_size,
            base_config.out_size,
            tanh;
            init_reservoir=init_reservoir,
            init_input=init_input,
            state_modifiers=base_config.state_modifiers,
            leak_coefficient=base_config.leak_coefficient,
            init_state=init_state,
            use_bias=true,
            init_bias=TestUtils.bias_initializer(base_config),
            readout_activation=base_config.readout_activation,)

    ps, st = ReservoirComputing.setup(rng, esn)

    ps, st = ReservoirComputing.train!(esn, input_data, target_data, ps, st,
                       base_config.training_method;
                       washout=base_config.washout,
                       solver=base_config.solver)

    return (ps, st, esn)
end

# ------------------------------------------------------------
# Legacy-checkpoint version compatibility
#
# ReservoirComputing.jl at some point renamed the `ps`/`st` NamedTuple field
# from `state_modifiers` (singular) to `states_modifiers` (plural) to match
# the `ESN` struct field name. Members trained under an older package
# version and loaded back under a newer one keep the old field name baked
# into the NamedTuple type. Worse than a simple rename: JLD2 also can't
# reconstruct the old `ESNCell`/`StatefulLayer`/closure types the current
# package expects, so it falls back to inert mock structs -- these fail
# `predict`/`collectstates`/every other generic entry point identically
# (`MethodError: no method matching apply(...)`, since the mock isn't a
# `LuxCore.AbstractLuxLayer`), not just `collectstates` as previously assumed.
# ------------------------------------------------------------

# 설치된 ReservoirComputing.jl이 `ps`/`st`에 쓰는 modifier 키. 0.12.24 등은
# `states_modifiers`, 0.12.32부터는 다시 `state_modifiers`라서 하드코딩하면 새 버전에서
# 방금 학습한 멤버까지 매번 "legacy"로 오판해 재마이그레이션한다.
const _RC_MODS_KEY = let
    _ps, _ = ReservoirComputing.setup(MersenneTwister(0),
        ESN(1, 4, 1; init_reservoir=rand_sparse(; sparsity=1.0)))
    :states_modifiers in propertynames(_ps) ? :states_modifiers : :state_modifiers
end

"""
    migrate_legacy_member(base_config, ps_old, st_old) -> (ps_new, st_new, esn_new)

`state_modifiers`(단수) 필드를 쓰는 옛 체크포인트 멤버를 지금 설치된
ReservoirComputing.jl이 기대하는 `states_modifiers`(복수) 구조로 옮겨 담는다.
새 reservoir dynamics를 만드는 게 아니다 -- `_train_one_member`와 완전히 같은
방식으로 골격(esn/ps/st)만 새로 만들고, 실제로 학습된 값(reservoir 가중치,
readout 가중치, 지금까지의 hidden state)만 그대로 옮긴다. `rng`/seed는 shape을
맞추는 용도일 뿐 아래에서 전부 덮어쓰므로 값 자체는 의미가 없다.

학습 데이터를 이 함수로 옮긴 멤버에 재생시켜 원래 학습 타깃 MSE가 1e-8 수준까지
그대로 재현됨을 확인했다 -- 가중치 전송이 손실 없이 이루어진다는 뜻이다.
"""
function migrate_legacy_member(base_config, ps_old, st_old)
    rng = MersenneTwister(0)

    init_reservoir = rand_sparse(; radius=base_config.radius, sparsity=base_config.sparsity)
    init_input = scaled_rand(; scaling=base_config.scaling)
    init_state = (rng, dims...) -> randn(rng, Float32, dims...)

    esn_new = ESN(base_config.in_size,
            base_config.res_size,
            base_config.out_size,
            tanh;
            init_reservoir=init_reservoir,
            init_input=init_input,
            state_modifiers=base_config.state_modifiers,
            leak_coefficient=base_config.leak_coefficient,
            init_state=init_state,
            use_bias=true,
            init_bias=TestUtils.bias_initializer(base_config),
            readout_activation=base_config.readout_activation)

    ps_new, st_new = ReservoirComputing.setup(rng, esn_new)

    ps_new.reservoir.input_matrix .= ps_old.reservoir.input_matrix
    ps_new.reservoir.reservoir_matrix .= ps_old.reservoir.reservoir_matrix
    ps_new.reservoir.bias .= ps_old.reservoir.bias
    ps_new.readout.weight .= ps_old.readout.weight

    st_new = merge(st_new, (reservoir = merge(st_new.reservoir, (carry = deepcopy(st_old.reservoir.carry),)),))

    return ps_new, st_new, esn_new
end

function train_ensemble(input_data::Matrix, target_data::Matrix, n_ensemble::Int;
                        base_config, vary_seed::Bool=true, seed_start::Int=1234)
    ensemble = []

    @showprogress desc="Training ensemble" for i in 1:n_ensemble
        seed = vary_seed ? seed_start + i - 1 : seed_start
        push!(ensemble, _train_one_member(input_data, target_data, base_config, seed))
    end

    return ensemble
end

# ------------------------------------------------------------
# Checkpointed variant: for large/expensive ensembles that need to survive
# a crash (OOM-kill etc.) without losing already-trained members, and that
# should never be retrained once cached on disk.
# ------------------------------------------------------------

_ckpt_member_path(checkpoint_dir, i) =
    joinpath(checkpoint_dir, "member_$(lpad(i, 6, '0')).jld2")

_ckpt_meta_path(checkpoint_dir) = joinpath(checkpoint_dir, "checkpoint_meta.jld2")

function _ckpt_fingerprint(base_config, seed_start::Int)
    fp = (
        seed_start=seed_start,
        in_size=base_config.in_size,
        res_size=base_config.res_size,
        out_size=base_config.out_size,
        radius=base_config.radius,
        sparsity=base_config.sparsity,
        scaling=base_config.scaling,
        leak_coefficient=base_config.leak_coefficient,
        washout=base_config.washout,
    )
    # 아래는 기본값이 아닐 때만 넣는다 (기존 체크포인트의 fingerprint를 그대로 유지)
    hasproperty(base_config, :bias_scaling) && base_config.bias_scaling != 0 && (fp = merge(fp, (bias_scaling=base_config.bias_scaling,)))
    isempty(base_config.state_modifiers) || (fp = merge(fp, (state_modifiers=string(base_config.state_modifiers),)))
    return fp
end

"""
    train_ensemble_checkpointed(
        input_data::Matrix,
        target_data::Matrix,
        n_ensemble::Int;
        base_config,
        vary_seed::Bool=true,
        seed_start::Int=1234,
        checkpoint_dir::AbstractString,
        on_progress::Function=(i, n) -> nothing,
    ) -> Vector

`train_ensemble`과 동일한 학습을 수행하지만, 멤버를 하나 학습할 때마다
`checkpoint_dir`에 `member_XXXXXX.jld2`로 즉시 저장한다 (tmp 파일에 쓴 뒤
`mv`로 최종 이름 교체 — 프로세스가 그 순간 죽어도 직전까지 저장된 멤버들은
손상되지 않는다. 긴 시간 하나의 JLD2 파일을 열어 두는 방식은 kill 시 파일
전체가 깨지므로 쓰지 않는다).

이미 `checkpoint_dir`에 학습된 멤버가 있으면(재시작 케이스) 그만큼은
재학습하지 않고 디스크에서 로드만 하고, 나머지만 이어서 학습한다. 이때
저장된 설정(`checkpoint_meta.jld2`)이 이번 호출의 `base_config`/`seed_start`와
다르면 서로 다른 실험이 섞이는 것을 막기 위해 에러를 던진다.

`on_progress(i, n_ensemble)`은 멤버 하나가 끝날 때마다 호출된다 (로깅/heartbeat
용도로 쓰라고 열어둔 훅 — 이 파일은 `experiments/` 실험 코드에 의존하지 않는다).
"""
function train_ensemble_checkpointed(
    input_data::Matrix, target_data::Matrix, n_ensemble::Int;
    base_config, vary_seed::Bool=true, seed_start::Int=1234,
    checkpoint_dir::AbstractString,
    on_progress::Function=(i, n) -> nothing,
    sparse::Bool=false,
)
    mkpath(checkpoint_dir)

    meta_path = _ckpt_meta_path(checkpoint_dir)
    fingerprint = _ckpt_fingerprint(base_config, seed_start)

    if isfile(meta_path)
        saved_fingerprint = JLD2.load(meta_path, "fingerprint")
        saved_fingerprint == fingerprint || throw(ArgumentError(
            "checkpoint_dir=\"$checkpoint_dir\" was trained with a different " *
            "base_config/seed_start than this call.\n  saved:   $saved_fingerprint\n" *
            "  current: $fingerprint\nUse a fresh checkpoint_dir if you intended to change the config."
        ))
    else
        jldsave(meta_path; fingerprint=fingerprint)
    end

    # Largest contiguous prefix already on disk (writes always happen in
    # increasing order, so a crash can only ever leave a clean prefix).
    n_done = 0
    while n_done < n_ensemble && isfile(_ckpt_member_path(checkpoint_dir, n_done + 1))
        n_done += 1
    end

    ensemble = Vector{Any}(undef, n_ensemble)
    n_migrated = 0
    for i in 1:n_done
        path = _ckpt_member_path(checkpoint_dir, i)
        # Portable export stores learned arrays/state, not session-specific closures.
        member = JLD2.jldopen(path,"r") do f
            if haskey(f,"ps") && haskey(f,"st")
                (f["ps"],f["st"],portable_esn(base_config))
            else
                f["member"]
            end
        end
        ps, st, esn = member

        if _RC_MODS_KEY in propertynames(ps)
            ensemble[i] = member
        else
            # Legacy checkpoint (see migrate_legacy_member docstring) -- migrate once and
            # resave in the current field layout so every later load is a plain fast read.
            migrated = migrate_legacy_member(base_config, ps, st)
            tmp_path = path * ".tmp"
            _jldsave_quiet(tmp_path; member=migrated)
            mv(tmp_path, path; force=true)
            ensemble[i] = migrated
            n_migrated += 1
        end
    end
    n_migrated > 0 && println(
        "train_ensemble_checkpointed: migrated $n_migrated legacy-format member(s) in " *
        "\"$checkpoint_dir\" to the current ReservoirComputing.jl field layout " *
        "(modifier field -> :$_RC_MODS_KEY)."
    )

    if n_done < n_ensemble
        @showprogress desc="Training ensemble (checkpointed)" for i in (n_done+1):n_ensemble
            seed = vary_seed ? seed_start + i - 1 : seed_start
            member = _train_one_member(input_data, target_data, base_config, seed; sparse=sparse)
            ensemble[i] = member

            final_path = _ckpt_member_path(checkpoint_dir, i)
            tmp_path = final_path * ".tmp"
            _jldsave_quiet(tmp_path; member)
            mv(tmp_path, final_path; force=true)

            on_progress(i, n_ensemble)
        end
    end

    return ensemble
end

# Reconstruct only the immutable architecture; every learned numerical value is
# read from the artifact. Initializers are not executed when forecasting.
portable_init_state(rng,dims...)=randn(rng,Float32,dims...)
function portable_esn(base)
    ESN(base.in_size,base.res_size,base.out_size,tanh;
        init_reservoir=rand_sparse(;radius=base.radius,sparsity=base.sparsity,return_sparse=true),
        init_input=scaled_rand(;scaling=base.scaling),init_state=portable_init_state,
        state_modifiers=base.state_modifiers,leak_coefficient=base.leak_coefficient,
        use_bias=true,init_bias=TestUtils.bias_initializer(base),readout_activation=base.readout_activation)
end
