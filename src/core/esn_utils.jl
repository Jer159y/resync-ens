module TestUtils

using FFTW
using JLD2
using LinearAlgebra
using Random
using Statistics
using ReservoirComputing
using ReservoirComputing: ESN, rand_sparse, scaled_rand, zeros32

export add_noise!
export build_base_config
export default_experiment_config
export bias_initializer
export readout_modifiers
export ExperimentConfig
export ks_features
export ks_score
export load_fixed_setting
export load_dataset
export rollout_autonomous
export train_single_esn
export warmup_teacher_forcing

function _pick_series_key(file)
    preferred = ("sol", "traj", "X", "u", "data")

    for key in preferred
        if haskey(file, key)
            value = read(file, key)
            if value isa AbstractVector
                value = reshape(value, 1, :)
            end
            if value isa AbstractMatrix
                return key, value
            end
        end
    end

    for key in keys(file)
        value = read(file, key)
        if value isa AbstractVector
            value = reshape(value, 1, :)
        end
        if value isa AbstractMatrix
            return key, value
        end
    end

    return nothing, nothing
end

function load_dataset(; data_path::AbstractString,
                        data_key::Union{Nothing, AbstractString}=nothing,
                        shift::Int=1000,
                        train_len::Int=5000,
                        valid_len::Int=0,
                        predict_len::Int=2000)
    sol = nothing
    h = nothing
    x = nothing

    JLD2.jldopen(data_path, "r") do file
        if data_key === nothing
            key, value = _pick_series_key(file)
            if key === nothing
                error("No 2D array found in JLD2 file: $(data_path). Available keys: $(collect(keys(file)))")
            end
            sol = value
        else
            if !haskey(file, data_key)
                error("Key $(data_key) not found in JLD2 file: $(data_path). Available keys: $(collect(keys(file)))")
            end
            sol = read(file, data_key)
            if !(sol isa AbstractMatrix)
                error("Key $(data_key) is not a 2D array in JLD2 file: $(data_path)")
            end
        end

        if haskey(file, "h")
            h = read(file, "h")
        end
        if haskey(file, "x")
            x = read(file, "x")
        end
    end

    total_len = size(sol, 2)
    needed_len = shift + train_len + valid_len + predict_len
    if total_len < needed_len
        error("Not enough time steps in $(data_path): need $(needed_len), got $(total_len)")
    end

    train_input = sol[:, shift:(shift + train_len - 1)]
    train_target = sol[:, (shift + 1):(shift + train_len)]

    valid_start = shift + train_len
    valid_input = sol[:, valid_start:(valid_start + valid_len - 1)]
    valid_target = sol[:, (valid_start + 1):(valid_start + valid_len)]

    test_start = shift + train_len + valid_len
    test_input = sol[:, test_start:(test_start + predict_len - 1)]
    test_target = sol[:, (test_start + 1):(test_start + predict_len)]
    initial_data = test_input[:, 1]

    return (sol=sol,
        h=h,
        x=x,
        train_input=train_input,
        train_target=train_target,
        valid_input=valid_input,
        valid_target=valid_target,
        test_input=test_input,
        test_target=test_target,
        initial_data=initial_data,
        input_data=train_input,
        target_data=train_target,
        test_data=test_input,
        initialdata=initial_data)
end

function load_fixed_setting(base_dir::AbstractString)
    @load joinpath(base_dir, "sms_traj.jld2") output_mean
    @load joinpath(base_dir, "roll_traj.jld2") trajectories
    @load joinpath(base_dir, "test_data.jld2") test_data
    @load joinpath(base_dir, "input_data.jld2") input_data

    return output_mean, trajectories, test_data, input_data
end

function add_noise!(input_data::AbstractArray;
                    noiselevel::Float64=exp10(-0.5),
                    rng::AbstractRNG=MersenneTwister(0))
    input_data .+= noiselevel .* randn(rng, size(input_data))
    return input_data
end

function build_base_config(; in_size::Int=128,
                            res_size::Int=1500,
                            out_size::Int=128,
                            radius::Float64=0.9,
                            sparsity::Float64=6 / 1500,
                            scaling::Float64=0.1,
                            leak_coefficient::Float64=0.1,
                            state_modifiers=(),
                            readout_activation=identity,
                            training_method,
                            washout::Int=1000,
                            steps::Int=2000,
                            solver)
    return (in_size=in_size,
            res_size=res_size,
            out_size=out_size,
            radius=radius,
            sparsity=sparsity,
            scaling=scaling,
            leak_coefficient=leak_coefficient,
            state_modifiers=state_modifiers,
            readout_activation=readout_activation,
            training_method=training_method,
            washout=washout,
            steps=steps,
            solver=solver)
end

struct ExperimentConfig
    in_size::Int
    res_size::Int
    out_size::Int
    radius::Float64
    sparsity::Float64
    scaling::Float64
    leak_coefficient::Float64
    state_modifiers
    readout_activation
    training_method
    washout::Int
    steps::Int
    solver
    bias_scaling::Float64          # reservoir bias ~ U(−b, b); 0 = 기존 (bias 0 → tanh ESN이 x → −x 대칭)
end

function default_experiment_config(; training_method,
                                    solver,
                                    in_size::Int=128,
                                    res_size::Int=1500,
                                    out_size::Int=128,
                                    radius::Float64=0.9,
                                    sparsity::Float64=6 / 1500,
                                    scaling::Float64=0.1,
                                    leak_coefficient::Float64=0.1,
                                    state_modifiers=(),
                                    readout_activation=identity,
                                    washout::Int=1000,
                                    steps::Int=2000,
                                    bias_scaling::Float64=0.0)
    return ExperimentConfig(
        in_size,
        res_size,
        out_size,
        radius,
        sparsity,
        scaling,
        leak_coefficient,
        state_modifiers,
        readout_activation,
        training_method,
        washout,
        steps,
        solver,
        bias_scaling
    )
end

"""
readout 앞 상태 변환 이름 → ESN `state_modifiers`. "linear" = 없음, "nlat1" = 홀수 번째 노드 제곱
(Pathak et al. 2017/2018의 r̃; KS/L96 재구성에 쓰인 대칭 깨기), "nlat2" = Chattopadhyay 2020 T₂.
"""
function readout_modifiers(name::AbstractString)
    name == "linear" && return ()
    name == "nlat1" && return (NLAT1,)
    name == "nlat2" && return (NLAT2,)
    throw(ArgumentError("unknown readout transform '$name' (linear | nlat1 | nlat2)"))
end

"""
reservoir bias 초기화: `bias_scaling` = 0이면 0 (기존 동작), 아니면 U(−b, b).
bias가 0이고 출력 변환이 없으면 tanh ESN의 닫힌 루프 사상은 x → −x에 대해 홀함수라서 그 대칭이 없는 계(L96, KS)를
재현할 수 없다 (Herteux & Räth 2020). bias나 NLAT1 같은 출력 변환이 대칭을 깬다.
"""
bias_initializer(base_config) = (b = hasproperty(base_config, :bias_scaling) ? Float64(base_config.bias_scaling) : 0.0;
    b == 0 ? zeros32 : (rng, dims...) -> Float32(b) .* (2 .* rand(rng, Float32, dims...) .- 1))

function train_single_esn(input_data, target_data, base_config, seed::Int)
    rng = MersenneTwister(seed)
    init_reservoir = rand_sparse(; radius=base_config.radius, sparsity=base_config.sparsity)
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
            init_bias=bias_initializer(base_config),
            readout_activation=base_config.readout_activation)

    ps, st = ReservoirComputing.setup(rng, esn)
    ps, st = ReservoirComputing.train!(esn, input_data, target_data, ps, st,
                        base_config.training_method;
                        washout=base_config.washout,
                        solver=base_config.solver)

    return esn, ps, st
end

function rollout_autonomous(ps, st, esn, steps::Int, initialdata)
    output, st_new = ReservoirComputing.predict(esn, steps, ps, st; initialdata=initialdata)
    return output, st_new
end

function warmup_teacher_forcing(ps, st, esn, data)
    _, st_new = ReservoirComputing.predict(esn, data, ps, st)
    return st_new
end

function power_spectrum(u::AbstractMatrix)
    Nx, Nt = size(u)
    spec = zeros(div(Nx, 2))

    for t in 1:Nt
        uhat = fft(u[:, t])
        e = abs2.(uhat[1:div(Nx, 2)])
        spec .+= e
    end

    spec ./= Nt
    return spec
end

function spectral_entropy(spec::AbstractVector)
    p = spec ./ sum(spec)
    eps = 1e-12
    H = -sum(p .* log.(p .+ eps))
    Hnorm = H / log(length(p))
    return Hnorm
end

function dominant_mode(spec::AbstractVector)
    return argmax(spec)
end

function highk_slope(spec::AbstractVector)
    n = length(spec)
    k0 = Int(round(0.7n))

    ks = collect(k0:n)
    ys = log.(spec[k0:n] .+ 1e-12)

    A = [ks ones(length(ks))]
    coef = A \ ys
    return coef[1]
end

function correlation_length(u::AbstractMatrix)
    Nx, Nt = size(u)
    c = zeros(Nx)

    for t in 1:Nt
        v = u[:, t] .- mean(u[:, t])
        for r in 1:Nx
            c[r] += mean(v .* circshift(v, r - 1))
        end
    end

    c ./= Nt
    c ./= c[1]

    idx = findfirst(x -> x < exp(-1), c)
    return idx === nothing ? Nx : idx
end

function lyapunov_proxy(u::AbstractMatrix)
    Nx, Nt = size(u)
    if Nt < 3
        return 0.0
    end

    growths = Float64[]
    for t in 1:(Nt - 2)
        d1 = norm(u[:, t + 1] - u[:, t])
        d2 = norm(u[:, t + 2] - u[:, t + 1])
        if d1 > 1e-12 && d2 > 1e-12
            push!(growths, log(d2 / d1))
        end
    end

    return mean(growths)
end

function defect_density(u::AbstractMatrix)
    Nx, Nt = size(u)
    counts = 0

    for t in 1:Nt
        v = u[:, t]
        for i in 2:(Nx - 1)
            if v[i] > v[i - 1] && v[i] > v[i + 1]
                counts += 1
            end
        end
    end

    return counts / (Nx * Nt)
end

function ks_features(u::AbstractMatrix)
    spec = power_spectrum(u)
    features = Dict{String, Float64}()
    features["spectral_entropy"] = spectral_entropy(spec)
    features["dominant_mode"] = dominant_mode(spec)
    features["highk_slope"] = highk_slope(spec)
    features["correlation_length"] = correlation_length(u)
    features["lyapunov_proxy"] = lyapunov_proxy(u)
    features["defect_density"] = defect_density(u)
    return features
end

function ks_score(f::Dict{String, Float64})
    score = 0.0
    score += clamp(f["spectral_entropy"], 0, 1)
    score += clamp(f["lyapunov_proxy"], 0, 1)
    score += clamp(-0.1 * f["highk_slope"], 0, 1)
    score += clamp(10 * f["defect_density"], 0, 1)
    return score / 4
end

end
