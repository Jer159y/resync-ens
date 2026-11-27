isdefined(@__MODULE__, :fidelity_names) || include(joinpath(@__DIR__, "..", "health", "fidelity_naming.jl"))
# ============================================================
# system_specs.jl
#
# Lorenz-96 / Kuramoto–Sivashinsky를 "member-fidelity reservoir ensemble" 파이프라인에 올리기 위한 시스템 정의.
# system_playground.jl(단일 reservoir 하이퍼파라미터 탐색 서버)과 R_system.jl(앙상블 파이프라인)이
# 같이 include한다 -- 두 곳이 같은 데이터 / 정규화 / ESN 설정 / 지표를 쓰도록 하는 단일 출처. top-level에서
# 아무것도 실행하지 않는다. src/resync_engine.jl을 먼저 include해야 한다 (TestUtils,
# train_ensemble_checkpointed/_train_one_member, atomic_save 등을 거기서 가져온다).
#
# 설정 (TOML, configs/<name>.toml; playground의 "Export"가 쓴다):
#   [system]      name ("lorenz96" | "ks") + 방정식/적분 파라미터 (L96: N, F / KS: N, L, ν) + dt(저장 간격),
#                 int_dt(적분 step), n_saved(데이터 길이), transient, seed
#   [data]        shift, train_len, normalize, noise
#   [esn]         res_size, degree(평균 연결 수; sparsity = degree/res_size), radius, leak, input_scaling, ridge,
#                 washout, seed (단일 reservoir seed = 앙상블 seed_start)
#   [experiment]  tau (0 = 자동: tau_lyap × Lyapunov time), tau_lyap, n_segments, n_ensemble, reserve, criterion, p,
#                 health_window (건강도 판정 창 W, τ와 무관한 절대 시간 블록; 0 = 자동 (≥ 8 LT인 τ의 최소 배수).
#                 W % τ == 0 또는 τ % W == 0)
#   [playground]  rollout_len, n_seeds (단일 reservoir 탐색 전용)
# 데이터는 [system]의 hash로 data/generated/<name>_<hash>.jld2에 캐시 (traj, ts, lyap = Benettin λ_max).
#
# 시스템 signature (질적 보조, 시스템마다 다름 -- 일반성 없음):
#   lorenz96  E1 에너지 E(t) = ½·mean_k x_k² 분포, E2 서쪽(−k) 진행파의 위상 속도(공간 이동 상관 최대 shift / 시간),
#             E3 공간 파수 스펙트럼(시간 평균 |FFT_k x|²) 형태
#   ks        K1 공간 파수 스펙트럼(피크 파수 ≈ 셀 크기), K2 스냅샷당 셀 수(u 극대 개수) 분포,
#             K3 u_x 분포(KS 특유의 비대칭 기울기: 가파른 앞면/완만한 뒷면)
# ============================================================

using TOML

const SYSTEM_CONFIG_DIR = normpath(joinpath(@__DIR__, "..", "..", "configs"))
const GENERATED_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "generated"))
const SUPPORTED_SYSTEMS = ["lorenz96", "ks", "duffing", "cml", "maooam", "cdv", "qg2", "jetlat", "file"]
const SPATIAL_SYSTEMS = ["lorenz96", "ks", "cml"]                # 격자 위의 장 (Hovmöller, 파수 스펙트럼, k–ω)
const REPO_DIR = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "duffing_signature.jl"))               # DUFFING_PARAMS, dwell_times, duffing_energy, duffing_lyapunov
include(joinpath(@__DIR__, "maooam.jl"))                          # MAOOAM tensor / RK4 / energies / field reconstruction
include(joinpath(@__DIR__, "cdv.jl"))                             # Charney–DeVore 6-mode model
include(joinpath(@__DIR__, "qg2.jl"))                             # two-layer QG (pyqg eddy / jet)

const SYSTEM_DEFAULTS = Dict(
    "lorenz96" => Dict(
        "system" => Dict{String,Any}("name" => "lorenz96", "N" => 20, "F" => 8.0, "dt" => 0.05, "int_dt" => 0.01,
            "n_saved" => 1_200_000, "transient" => 20_000, "seed" => 1, "obs_N" => 0),
        "data" => Dict{String,Any}("shift" => 1000, "train_len" => 20_000, "normalize" => true, "noise" => 1e-6),
        "esn" => Dict{String,Any}("res_size" => 3000, "degree" => 6.0, "radius" => 0.5, "leak" => 0.5, "input_scaling" => 0.5,
            "ridge" => 1e-5, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.0, "readout" => "nlat1", "sparse" => true),
        "experiment" => Dict{String,Any}("tau" => 0, "tau_lyap" => 5.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 100,
            "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
        "playground" => Dict{String,Any}("rollout_len" => 10_000, "n_seeds" => 3)),
    "ks" => Dict(
        "system" => Dict{String,Any}("name" => "ks", "N" => 32, "L" => 22.0, "nu" => 1.0, "dt" => 0.25, "int_dt" => 0.05,
            "n_saved" => 1_200_000, "transient" => 4_000, "seed" => 1, "obs_N" => 0, "damp" => 0.0),
        "data" => Dict{String,Any}("shift" => 1000, "train_len" => 20_000, "normalize" => true, "noise" => 1e-6),
        "esn" => Dict{String,Any}("res_size" => 3000, "degree" => 6.0, "radius" => 0.5, "leak" => 0.5, "input_scaling" => 0.5,
            "ridge" => 1e-5, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.0, "readout" => "nlat1", "sparse" => true),
        "experiment" => Dict{String,Any}("tau" => 0, "tau_lyap" => 5.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 100,
            "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
        "playground" => Dict{String,Any}("rollout_len" => 20_000, "n_seeds" => 3)),
    # Duffing (강제 진동, 2D 비공간계): 데이터는 생성하지 않고 기존 파일을 쓴다 (StatementI / R 실험과 같은 truth).
    # 방정식 ẍ + δẋ − αx + βx³ = γ cos ωt (duffing_signature.jl의 Holmes 설정과 같아야 한다).
    "duffing" => Dict(
        "system" => Dict{String,Any}("name" => "duffing", "N" => 2, "dt" => 0.1, "int_dt" => 0.001, "alpha" => 1.0, "beta" => 1.0,
            "delta" => 0.3, "gamma" => 0.4, "omega" => 1.2, "data_file" => "data/duffing/duffing_oscillator.jld2",
            "n_saved" => 0, "transient" => 0, "seed" => 1),
        "data" => Dict{String,Any}("shift" => 1000, "train_len" => 5000, "normalize" => false, "noise" => 1e-6, "reference_len" => 3_000_000),
        "esn" => Dict{String,Any}("res_size" => 200, "degree" => 6.0, "radius" => 0.9, "leak" => 0.3, "input_scaling" => 0.1,
            "ridge" => 1e-6, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.0, "readout" => "linear", "sparse" => false),
        "experiment" => Dict{String,Any}("tau" => 600, "tau_lyap" => 0.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 150,
            "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 600),
        "playground" => Dict{String,Any}("rollout_len" => 20_000, "n_seeds" => 3)),
)
# Chaté–Manneville coupled map lattice (spatiotemporal intermittency, 2026-10-21): tent map of slope r on [0, 1],
# f(x) = x above 1 (laminar, absorbing: a laminar site turns turbulent again only through its neighbours), diffusive
# coupling ε. At r = 3, ε = 0.4 the N = 64 truth stays turbulent for 6e5 steps (N = 32 laminarises), while a surrogate that
# laminarises cannot recover — the "failure does not heal by itself" regime resync needs. dt = 1 map step.
SYSTEM_DEFAULTS["cml"] = Dict(
    "system" => Dict{String,Any}("name" => "cml", "N" => 64, "r" => 3.0, "eps" => 0.4, "dt" => 1.0, "int_dt" => 1.0,
        "n_saved" => 600_000, "transient" => 10_000, "seed" => 1),
    "data" => Dict{String,Any}("shift" => 1000, "train_len" => 20_000, "normalize" => true, "noise" => 1e-6),
    "esn" => Dict{String,Any}("res_size" => 1000, "degree" => 6.0, "radius" => 0.5, "leak" => 0.5, "input_scaling" => 0.5,
        "ridge" => 1e-5, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.0, "readout" => "nlat1", "sparse" => true),
    "experiment" => Dict{String,Any}("tau" => 0, "tau_lyap" => 5.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 100,
        "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
    "playground" => Dict{String,Any}("rollout_len" => 20_000, "n_seeds" => 3))
# MAOOAM (2026-10-28): 36-variable coupled ocean–atmosphere QG model (maooam.jl), physical parameters = the MAOOAM v1.0
# paper set DDV2016 (strong coupling d = 1.1e-7 → coupled low-frequency variability). Time unit 1/f0 ≈ 0.112 day; all
# MAOOAM_PARAM_KEYS are [system] keys (configs/maooam.toml = DDV2016, configs/candidates/maooam_vl2016.toml = weak coupling).
# N = state dimension (must equal 2·natm + 2·noc for the atm_x/atm_y/oc_x/oc_y mode selection). Non-spatial.
# Spin-up: the ocean streamfunction relaxes on M(r' + d')/(M + G) ≈ 5e-7 per unit (~2e6 units ≈ 600 yr); from small
# random states the model first sits near a weakly unstable steady state and becomes chaotic only after ~3e6–1e7 units
# (the MAOOAM papers use 1e8). transient = 2e6 saved steps × dt 5 = 1e7 units. dt = 5 units ≈ 0.56 day.
# DDV2016 on the attractor (1e7 units after spin-up): λ_max ≈ 0.0115 per unit (LT ≈ 87 units ≈ 9.7 days ≈ 17 steps), atmosphere
# e-fold ACF ≈ 3 days, ocean ψ_o ≈ 15 yr, LFV periods ≈ 41.5 yr (To2, ψa1) / 83 yr (ψo2) → train_len 1e5 steps ≈ 150 yr ≈ 3.6 cycles.
SYSTEM_DEFAULTS["maooam"] = Dict(
    "system" => merge(Dict{String,Any}("name" => "maooam", "N" => 36, "dt" => 5.0, "int_dt" => 0.1,
        "n_saved" => 1_200_000, "transient" => 2_000_000, "seed" => 1), MAOOAM_PARAM_SETS["DDV2016"]),
    "data" => Dict{String,Any}("shift" => 1000, "train_len" => 100_000, "normalize" => true, "noise" => 1e-6, "reference_len" => 500_000),
    # learn_check (2026-10-28): radius 0.3 / input 0.1 / leak 1.0 → first-step RMSE 0.045, VPT 0.7–0.9 LT (radius 0.5 / leak 0.5: 0.2, 0.2 LT)
    "esn" => Dict{String,Any}("res_size" => 1000, "degree" => 6.0, "radius" => 0.3, "leak" => 1.0, "input_scaling" => 0.1,
        "ridge" => 1e-8, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.5, "readout" => "nlat1", "sparse" => true),
    "experiment" => Dict{String,Any}("tau" => 0, "tau_lyap" => 5.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 100,
        "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
    "playground" => Dict{String,Any}("rollout_len" => 20_000, "n_seeds" => 3))
# Charney–DeVore (2026-10-28, cdv.jl): 6-mode barotropic channel flow over topography, Crommelin et al. 2004 regime setting
# (zonal ↔ blocked transitions in the x1–x4 plane). Time unit ≈ 1 day; λ_max ≈ 0.023 → LT ≈ 43 units = 43 steps at dt = 1.
SYSTEM_DEFAULTS["cdv"] = Dict(
    "system" => Dict{String,Any}("name" => "cdv", "N" => 6, "dt" => 1.0, "int_dt" => 0.05, "x1s" => 0.95, "x4s" => -0.76095,
        "C" => 0.1, "beta" => 1.25, "gamma" => 0.2, "b" => 0.5, "n_saved" => 1_200_000, "transient" => 10_000, "seed" => 1),
    "data" => Dict{String,Any}("shift" => 1000, "train_len" => 20_000, "normalize" => true, "noise" => 1e-6, "reference_len" => 500_000),
    "esn" => Dict{String,Any}("res_size" => 500, "degree" => 6.0, "radius" => 0.5, "leak" => 0.5, "input_scaling" => 0.5,
        "ridge" => 1e-6, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.5, "readout" => "nlat1", "sparse" => true),
    "experiment" => Dict{String,Any}("tau" => 0, "tau_lyap" => 5.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 100,
        "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
    "playground" => Dict{String,Any}("rollout_len" => 20_000, "n_seeds" => 3))
# Two-layer QG (2026-10-28, qg2.jl): pyqg QGModel, doubly periodic, eddy (default) / jet sets. The state is the low-wavenumber
# Fourier coefficients of ψ1, ψ2 (|k|, |l| ≤ obs_k; N = 8·obs_k·(obs_k + 1)); diagnostics on the full grid (energies, ū1(y),
# q1 snapshots, KE spectrum) are stored next to the trajectory. Time unit = seconds; dt = 1 day, int_dt = 1 h (pyqg dt).
# Measured (5 / 10 yr spin-up): eddy λ ≈ 0.036 per day (LT ≈ 28 days), jet λ ≈ 0.017 (LT ≈ 59 days, 3–4 zonal jets). obs_k = 4
# holds only 10 % (eddy) / 35 % (jet) of the KE (eddies peak at k ≈ 5–10) → obs_k = 8 (576 values); configs add reduce = "pod".
SYSTEM_DEFAULTS["qg2"] = Dict(
    "system" => merge(Dict{String,Any}("name" => "qg2", "N" => 576, "dt" => 86400.0, "int_dt" => 3600.0,
        "n_saved" => 120_000, "transient" => 1825, "seed" => 1), QG2_PARAM_SETS["eddy"], Dict{String,Any}("obs_k" => 8)),
    "data" => Dict{String,Any}("shift" => 1000, "train_len" => 20_000, "normalize" => true, "noise" => 1e-6, "reference_len" => 80_000),
    "esn" => Dict{String,Any}("res_size" => 1000, "degree" => 6.0, "radius" => 0.5, "leak" => 0.5, "input_scaling" => 0.5,
        "ridge" => 1e-6, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.5, "readout" => "nlat1", "sparse" => true),
    "experiment" => Dict{String,Any}("tau" => 0, "tau_lyap" => 5.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 100,
        "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
    "playground" => Dict{String,Any}("rollout_len" => 20_000, "n_seeds" => 3))
# JetLat (2026-10-28): observed North Atlantic winter jet (Strommen et al. 2023): jet latitude, PC1, PC2 of 850 hPa u, ERA-20C
# DJF 1901–2010, 9810 days (tools/fetch_jetlat.jl). No equations: no flow map / true λ; lyap = 1 / e-fold ACF time of the jet
# latitude (≈ 7.75 days) is only a time unit. Split: 3000 train, 150 × 10 test, ~5300 reference days (R_system needs ≥ 5000).
SYSTEM_DEFAULTS["jetlat"] = Dict(
    "system" => Dict{String,Any}("name" => "jetlat", "N" => 3, "dt" => 1.0, "int_dt" => 1.0,
        "data_file" => "data/generated/jetlat/jetlat_era20c.jld2", "n_saved" => 0, "transient" => 0, "seed" => 1),
    "data" => Dict{String,Any}("shift" => 1, "train_len" => 3000, "normalize" => true, "noise" => 1e-6, "reference_len" => 5300),
    "esn" => Dict{String,Any}("res_size" => 300, "degree" => 6.0, "radius" => 0.5, "leak" => 0.5, "input_scaling" => 0.5,
        "ridge" => 1e-4, "washout" => 100, "seed" => 1234, "bias_scaling" => 0.5, "readout" => "nlat1", "sparse" => true),
    "experiment" => Dict{String,Any}("tau" => 10, "tau_lyap" => 0.0, "n_segments" => 150, "n_ensemble" => 300, "reserve" => 100,
        "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
    "playground" => Dict{String,Any}("rollout_len" => 1500, "n_seeds" => 3))
# "file": any system generated elsewhere (tools/make_probe_data.jl) into a JLD2 with traj (D × T), ts, lyap (per time
# unit). For screening candidate systems with tools/health_eval.jl / ensemble_probe.jl; no flow map or system signatures.
SYSTEM_DEFAULTS["file"] = Dict(
    "system" => Dict{String,Any}("name" => "file", "data_file" => "", "N" => 0, "dt" => 1.0, "int_dt" => 1.0, "spatial" => true,
        "label" => ""),
    "data" => Dict{String,Any}("shift" => 1000, "train_len" => 20_000, "normalize" => true, "noise" => 1e-6),
    "esn" => Dict{String,Any}("res_size" => 500, "degree" => 6.0, "radius" => 0.4, "leak" => 0.9, "input_scaling" => 0.1,
        "ridge" => 1e-3, "washout" => 500, "seed" => 1234, "bias_scaling" => 0.9, "readout" => "nlat2", "sparse" => true),
    "experiment" => Dict{String,Any}("tau" => 0, "tau_lyap" => 5.0, "n_segments" => 200, "n_ensemble" => 300, "reserve" => 100,
        "criterion" => "feature_distance_from_centroid", "p" => 0.10, "health_window" => 0),
    "playground" => Dict{String,Any}("rollout_len" => 20_000, "n_seeds" => 3))
SYSTEM_DEFAULTS["file"]["data"]["reference_len"] = 100_000
for name in ("lorenz96", "ks", "cml")
    SYSTEM_DEFAULTS[name]["data"]["reference_len"] = 500_000       # test 뒤 긴 truth (건강도 보정 · 다양성 null · dynamics null)
end
# 상태 축소 (2026-10-28): reduce = "none" | "pod" | "fourier", reduce_k = 모드 수. 기본값은 hash에서 빠진다 (기존 이름 유지).
for (_, c) in SYSTEM_DEFAULTS
    c["system"]["reduce"] = "none"; c["system"]["reduce_k"] = 0
    c["experiment"]["mahal_var_frac"] = 1.0       # R0b L0 분산 비율 (1.0 = 전체 공분산); 기본값이 없으면 merge_config가 TOML 값을 버린다
end

"필드 설명 (playground 폼 툴팁)."
const FIELD_HELP = Dict(
    "data.reference_len" => "test 구간 뒤 긴 truth 기준 궤적 길이 (step): 다양성 null, 건강도 보정, dynamics null band",
    "system.data_file" => "Duffing: 데이터 파일 (repo 기준 경로; 생성하지 않음)",
    "system.alpha" => "Duffing α (선형 복원 −αx)", "system.beta" => "Duffing β (x³ 항)", "system.delta" => "Duffing 감쇠 δ",
    "system.gamma" => "Duffing 외력 진폭 γ", "system.omega" => "Duffing 외력 각진동수 ω",
    "experiment.health_window" => "건강도 판정 창 W (step; τ와 무관, 시스템마다 정함). 0 = 자동: 8 Lyapunov time 이상인 τ의 가장 작은 배수. W와 τ는 한쪽이 다른 쪽의 배수",
    "system.atm_x" => "MAOOAM 대기 x 모드 블록 수 (M = 1..atm_x)", "system.atm_y" => "MAOOAM 대기 y 모드 (P = 1..atm_y)",
    "system.oc_x" => "MAOOAM 해양 x 모드 (반정수 파수 H/2, H = 1..oc_x)", "system.oc_y" => "MAOOAM 해양 y 모드 (P = 1..oc_y)",
    "system.d" => "MAOOAM 해양–대기 기계적 결합 d (s⁻¹; 1.1e-7 = DDV2016 LFV, 1e-8 = 약결합)",
    "system.x1s" => "CDV 띠 흐름 강제 x1* (0.95)", "system.x4s" => "CDV 강제 x4* (−0.76095)", "system.C" => "CDV 감쇠 C (0.1 ≈ 10일)",
    "system.gamma" => "CDV 지형 진폭 γ (0.2 ≈ 200 m)", "system.b" => "CDV 채널 종횡비 b (0.5)",
    "system.reduce" => "상태 축소: none | pod (training 창의 POD/EOF 선도 reduce_k 모드) | fourier (1차원 주기 격자의 파수 0..reduce_k, 2k+1 차원)",
    "system.reduce_k" => "축소 모드 수 (pod: 차원 = k, fourier: 최대 파수 k)",
    "experiment.mahal_var_frac" => "R0b L0 (blowup) Mahalanobis를 분산의 이 비율을 설명하는 상위 주성분에서만 (1.0 = 전체 공분산, 기존 동작)",
    "system.obs_k" => "QG2 관측 최대 파수 (|k|, |l| ≤ obs_k의 ψ1, ψ2 Fourier 계수; N = 8·obs_k·(obs_k+1))",
    "system.N" => "격자 수 (L96: 변수 개수 = 상태 차원, KS: 공간 격자 = 상태 차원)",
    "system.obs_N" => "KS / L96 관측 격자 수 (0 = N). 적분은 N 격자로 하고 N/obs_N 간격 지점만 ESN 상태로 쓴다 (N의 약수). L=22에서는 16이면 Fourier 보간 복원 오차 ≈ 2%",
    "system.F" => "L96 외력 F (8 = 표준 카오스)",
    "system.L" => "KS 도메인 길이 (22 = 표준, 클수록 셀 많고 고차원 카오스)",
    "system.nu" => "KS 4차 확산 계수 ν",
    "system.dt" => "저장 간격 (물리 시간) = reservoir 한 step",
    "system.int_dt" => "적분 step (dt의 약수여야 함)",
    "system.n_saved" => "생성할 데이터 길이 (저장 step 수)",
    "system.transient" => "버리는 초기 과도 (저장 step 수)",
    "data.train_len" => "training 길이 (step)", "data.shift" => "데이터 앞부분 건너뛰기 (step)",
    "data.normalize" => "training target의 차원별 평균/표준편차로 표준화",
    "data.noise" => "training 입력 noise 크기",
    "esn.degree" => "reservoir 행렬 평균 연결 수 (sparsity = degree / res_size)",
    "esn.bias_scaling" => "reservoir bias ~ U(−b, b). 0이고 readout = linear이면 ESN이 x → −x 대칭이 되어 그 대칭이 없는 계(L96, KS)를 재현할 수 없다",
    "esn.sparse" => "reservoir 행렬을 SparseMatrixCSC로 저장 (값은 같고 결과 차이 ~1e-12). 큰 reservoir / 많은 멤버에서 크게 빠름. false면 hash에서 빠져 기존 디렉터리 유지",
    "esn.readout" => "readout 앞 상태 변환: linear | nlat1 (홀수 번째 노드 제곱, Pathak 2017/2018) | nlat2. 대칭 깨기",
    "esn.seed" => "reservoir seed (앙상블 member i = seed + i − 1; playground의 n_seeds도 여기서부터)",
    "experiment.tau" => "resync segment 길이 τ (step). 0 = 자동 (tau_lyap × Lyapunov time)",
    "experiment.tau_lyap" => "tau = 0일 때 τ = tau_lyap / (λ_max·dt)",
    "playground.rollout_len" => "단일 reservoir 자율 rollout 길이 (step)",
    "playground.n_seeds" => "비교할 seed 개수 (seed, seed+1, …) — 같은 설정의 seed 편차 확인용",
)

deepcopy_config(c) = Dict(k => copy(v) for (k, v) in c)
default_config(name::AbstractString) = deepcopy_config(SYSTEM_DEFAULTS[name])

"기본값 위에 덮어쓰기 (없는 섹션/키는 기본값 유지, 타입은 기본값 타입으로 변환)."
function merge_config(name::AbstractString, over::AbstractDict)
    cfg = default_config(name)
    for (sec, d) in over
        haskey(cfg, sec) || continue
        for (k, v) in d
            haskey(cfg[sec], k) || continue
            cfg[sec][k] = coerce_like(cfg[sec][k], v)
        end
    end
    return cfg
end

coerce_like(ref::Bool, v) = v isa Bool ? v : lowercase(string(v)) in ("true", "1", "on", "yes")
coerce_like(ref::Integer, v) = v isa Integer ? Int(v) : round(Int, parse(Float64, string(v)))
coerce_like(ref::AbstractFloat, v) = v isa Real ? Float64(v) : parse(Float64, string(v))
coerce_like(ref::AbstractString, v) = string(v)

load_config(path::AbstractString) = (d = fidelity_names(TOML.parsefile(path)); merge_config(d["system"]["name"], d))
function save_config(path::AbstractString, cfg; header::AbstractString="")
    mkpath(dirname(path))
    open(path, "w") do io
        isempty(header) || println(io, join(["# " * l for l in split(header, '\n')], "\n"))
        TOML.print(io, cfg; sorted=true)
    end
end

"KS 관측 격자 수 (obs_N = 0이면 적분 격자 N 그대로)."
obs_dim(cfg) = (M = Int(get(cfg["system"], "obs_N", 0)); M > 0 ? M : cfg["system"]["N"])
"ESN / 평가가 보는 상태 차원 (KS는 관측 격자, 나머지는 N)."
state_dim(cfg) = (r = get(cfg["system"], "reduce", "none"); k = Int(get(cfg["system"], "reduce_k", 0));
    r == "pod" ? k : r == "fourier" ? 2k + 1 : obs_dim(cfg))
"관측 격자 = 적분 격자의 stride 간격 부분집합 (stride = N / obs_N)."
function obs_stride(cfg)
    N, M = cfg["system"]["N"], obs_dim(cfg)
    N % M == 0 || error("system.obs_N = $M must divide system.N = $N")
    return N ÷ M
end
# obs_N = 0 / damp = 0(추가 전 동작)은 hash에서 빼서 기존 데이터 / 앙상블 이름 유지; 적분 데이터 hash는 obs_N과 무관
"""
Data-file paths as they were before the 2026-10-21 layout change (sample_data/ → data/). Only used inside hashes, so run
directories and ensemble names made before the move keep their names (Duffing and "file" systems hash `data_file`).
"""
legacy_path(s::AbstractString) = replace(s, "data/duffing/" => "sample_data/v1/data/", r"^data/generated/" => "sample_data/generated/")
legacy_path(x) = x
hash_system(sys) = Dict(k => (k == "data_file" ? legacy_path(v) : v) for (k, v) in sys if !(k == "reduce" && v == "none") && !(k == "reduce_k" && v == 0))
system_hash(cfg) = string(hash(sort(collect(filter(kv -> kv != ("obs_N" => 0) && kv != ("damp" => 0.0), hash_system(cfg["system"]))); by=first)); base=16)
full_data_hash(cfg) = string(hash(sort(collect(filter(kv -> !(first(kv) in ("obs_N", "reduce", "reduce_k")) && kv != ("damp" => 0.0), cfg["system"])); by=first)); base=16)
full_data_path(cfg) = joinpath(GENERATED_DATA_DIR, "$(cfg["system"]["name"])_$(full_data_hash(cfg)).jld2")
"ESN이 읽는 데이터 파일: 관측 격자를 쓰면 적분 격자 파일에서 뽑은 별도 파일 (R0a도 이 경로를 직접 load한다)."
data_path(cfg) = cfg["system"]["name"] in ("duffing", "file", "jetlat") ? joinpath(REPO_DIR, cfg["system"]["data_file"]) :
    obs_stride(cfg) == 1 ? full_data_path(cfg) : replace(full_data_path(cfg), r"\.jld2$" => "_obs$(obs_dim(cfg)).jld2")
is_spatial(cfg) = get(cfg["system"], "reduce", "none") == "none" &&
    (cfg["system"]["name"] in SPATIAL_SYSTEMS || (cfg["system"]["name"] == "file" && cfg["system"]["spatial"]))
"축소 데이터 파일 (기준 = data_path; POD는 training 창에 맞추므로 shift / train_len이 이름에 들어간다)."
function reduced_data_path(cfg)
    r = cfg["system"]["reduce"]; k = cfg["system"]["reduce_k"]; b = data_path(cfg)
    r == "pod" && return replace(b, r"\.jld2$" => "_pod$(k)_s$(cfg["data"]["shift"])_t$(cfg["data"]["train_len"]).jld2")
    r == "fourier" && return replace(b, r"\.jld2$" => "_fourier$(k).jld2")
    error("unknown system.reduce = $r (none | pod | fourier)")
end
"""
비공간계 위상도 축 (state index 2개 + 라벨). Duffing (x, ẋ); CDV x1–x4 (Crommelin 2004의 레짐 평면); MAOOAM = 해양 ψ_o,2 – δT_o,2 평면 (결합 LFV 고리가 보이는 평면,
Vannitsem et al. 2015; 두 번째 해양 모드 φ(1, 2)).
"""
function phase_dims(cfg)
    r = get(cfg["system"], "reduce", "none")
    r == "pod" && return (1, 2), ("PC1", "PC2")
    r == "fourier" && return (2, 3), ("Re ĉ1", "Im ĉ1")
    cfg["system"]["name"] == "jetlat" && return (2, 3), ("PC1 (u850)", "PC2 (u850)")
    cfg["system"]["name"] == "qg2" && return (1, 2), ("Re ψ1(0,1)", "Im ψ1(0,1)")
    cfg["system"]["name"] == "cdv" && return (1, 4), ("x1 (zonal flow)", "x4")
    if cfg["system"]["name"] == "maooam"
        m = maooam_from_cfg(cfg["system"]); b = maooam_blocks(m)
        return (b.psi_o[2], b.T_o[2]), ("ψo2", "To2")
    end
    return (1, 2), ("x1", "x2")
end

"Duffing 방정식의 RK4 한 step (물리 좌표, 시각 t)."
function duffing_rk4(u::AbstractVector, t::Real, h::Real, p)
    f(x, τ) = (x[2], -p["delta"] * x[2] + p["alpha"] * x[1] - p["beta"] * x[1]^3 + p["gamma"] * cos(p["omega"] * τ))
    k1 = f(u, t); a = (u[1] + h / 2 * k1[1], u[2] + h / 2 * k1[2])
    k2 = f(a, t + h / 2); b = (u[1] + h / 2 * k2[1], u[2] + h / 2 * k2[2])
    k3 = f(b, t + h / 2); c = (u[1] + h * k3[1], u[2] + h * k3[2])
    k4 = f(c, t + h)
    return [u[1] + h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1]), u[2] + h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])]
end

# ------------------------------------------------------------
# 1. 적분기
# ------------------------------------------------------------

function l96_rhs!(du, u, F)
    N = length(u)
    @inbounds for i in 1:N
        du[i] = (u[mod1(i + 1, N)] - u[mod1(i - 2, N)]) * u[mod1(i - 1, N)] - u[i] + F
    end
end

function l96_rk4!(x, dt, F, k1, k2, k3, k4, tmp)
    l96_rhs!(k1, x, F); tmp .= x .+ (dt / 2) .* k1
    l96_rhs!(k2, tmp, F); tmp .= x .+ (dt / 2) .* k2
    l96_rhs!(k3, tmp, F); tmp .= x .+ dt .* k3
    l96_rhs!(k4, tmp, F); x .+= (dt / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

"Kassam–Trefethen ETDRK4 (u_t = −u u_x − u_xx − ν u_xxxx, 주기 [0, L)); 실수 FFT, Nyquist 파수 0, 계수는 contour 적분 (M=32)."
function ks_etdrk4_setup(N::Int, L::Real, ν::Real, h::Real; damp::Real=0.0)
    k = (2π / L) .* vcat(0:(N ÷ 2 - 1), 0)
    Lop = k .^ 2 .- ν .* k .^ 4 .- damp                  # damp > 0: damped KS (u = 0 absorbing; spatiotemporal intermittency)
    E = exp.(h .* Lop); E2 = exp.(h .* Lop ./ 2)
    M = 32
    r = exp.(im .* π .* ((1:M) .- 0.5) ./ M)
    LR = h .* Lop .+ transpose(r)
    Q = h .* real.(vec(mean((exp.(LR ./ 2) .- 1) ./ LR; dims=2)))
    f1 = h .* real.(vec(mean((-4 .- LR .+ exp.(LR) .* (4 .- 3 .* LR .+ LR .^ 2)) ./ LR .^ 3; dims=2)))
    f2 = h .* real.(vec(mean((2 .+ LR .+ exp.(LR) .* (-2 .+ LR)) ./ LR .^ 3; dims=2)))
    f3 = h .* real.(vec(mean((-4 .- 3 .* LR .- LR .^ 2 .+ exp.(LR) .* (4 .- LR)) ./ LR .^ 3; dims=2)))
    g = -0.5im .* k
    return (E=E, E2=E2, Q=Q, f1=f1, f2=f2, f3=f3, g=g, N=N)
end

ks_nl(v, c) = c.g .* rfft(irfft(v, c.N) .^ 2)
function ks_step(v, c)
    Nv = ks_nl(v, c)
    a = c.E2 .* v .+ c.Q .* Nv; Na = ks_nl(a, c)
    b = c.E2 .* v .+ c.Q .* Na; Nb = ks_nl(b, c)
    cc = c.E2 .* a .+ c.Q .* (2 .* Nb .- Nv); Nc = ks_nl(cc, c)
    return c.E .* v .+ Nv .* c.f1 .+ 2 .* (Na .+ Nb) .* c.f2 .+ Nc .* c.f3
end
ks_fwd(u) = rfft(u)
ks_inv(v, c) = irfft(v, c.N)

"""
    generate_system_data(cfg) -> (traj, ts, lyap)

[system] 설정으로 궤적 생성 (transient 제거) + Benettin 두-궤적 재정규화로 λ_max 추정 (저장 step마다 재정규화,
처음 min(n_saved, 20000) 저장 step 사용; 단위 = 1/물리시간).
"""
function generate_system_data(cfg)
    p = cfg["system"]
    name = p["name"]
    dt = p["dt"]; h = p["int_dt"]
    sub = round(Int, dt / h)
    @assert isapprox(sub * h, dt; rtol=1e-9) "int_dt must divide dt"
    n = p["n_saved"]; ntr = p["transient"]
    rng = MersenneTwister(p["seed"])
    D = p["N"]
    traj = Matrix{Float64}(undef, D, n)
    nly = min(n, 20_000); d0 = 1e-8; acc = 0.0
    extra = (;)
    if name == "lorenz96"
        F = p["F"]
        x = F .+ 0.01 .* randn(rng, D)
        bufs = [zeros(D) for _ in 1:5]
        for _ in 1:(ntr * sub)
            l96_rk4!(x, h, F, bufs...)
        end
        y = x .+ d0 .* normalize(randn(rng, D))
        for t in 1:n
            for _ in 1:sub
                l96_rk4!(x, h, F, bufs...)
                t <= nly && l96_rk4!(y, h, F, bufs...)
            end
            traj[:, t] .= x
            if t <= nly
                d = norm(y .- x); acc += log(d / d0); y .= x .+ (d0 / d) .* (y .- x)
            end
        end
    elseif name == "ks"
        c = ks_etdrk4_setup(D, p["L"], p["nu"], h; damp=get(p, "damp", 0.0))
        xs = (0:(D - 1)) .* (p["L"] / D)
        u0 = cos.(2π .* xs ./ p["L"]) .* (1 .+ sin.(2π .* xs ./ p["L"])) .+ 0.01 .* randn(rng, D)
        v = ks_fwd(u0)
        for _ in 1:(ntr * sub)
            v = ks_step(v, c)
        end
        w = ks_fwd(ks_inv(v, c) .+ d0 .* normalize(randn(rng, D)))
        for t in 1:n
            for _ in 1:sub
                v = ks_step(v, c)
                t <= nly && (w = ks_step(w, c))
            end
            u = ks_inv(v, c)
            traj[:, t] .= u
            if t <= nly
                uw = ks_inv(w, c)
                d = norm(uw .- u); acc += log(d / d0); w = ks_fwd(u .+ (d0 / d) .* (uw .- u))
            end
        end
    elseif name == "maooam"
        m = maooam_from_cfg(p)
        m.ndim == D || error("maooam: [system] N = $D but the mode selection gives $(m.ndim) variables")
        x = 0.01 .* randn(rng, D)
        bufs = [zeros(D) for _ in 1:5]; bufy = [zeros(D) for _ in 1:5]
        for _ in 1:(ntr * sub)
            maooam_rk4!(x, h, m, bufs...)
        end
        y = x .+ d0 .* normalize(randn(rng, D))
        for t in 1:n
            for _ in 1:sub
                maooam_rk4!(x, h, m, bufs...)
                t <= nly && maooam_rk4!(y, h, m, bufy...)
            end
            traj[:, t] .= x
            if t <= nly
                d = norm(y .- x); acc += log(d / d0); y .= x .+ (d0 / d) .* (y .- x)
            end
        end
    elseif name == "qg2"
        m = qg2_from_cfg(p)
        m.nobs == D || error("qg2: [system] N = $D but obs_k = $(p["obs_k"]) gives $(m.nobs) values")
        qh = qg2_initial(m, rng)
        for _ in 1:(ntr * sub)
            qh = qg2_step(qh, h, m)
        end
        nrm(a, b) = norm(qg2_invert(a .- b, m))
        pq = qg2_pv(cat(randn(rng, ComplexF64, size(qh, 1), size(qh, 2)), zeros(ComplexF64, size(qh, 1), size(qh, 2)); dims=3), m)
        dq = 1e-6 * norm(qg2_invert(qh, m)); yq = qh .+ pq .* (dq / nrm(qh .+ pq, qh))
        nk = m.nx ÷ 2 + 1
        kidx = [round(Int, sqrt((i - 1)^2 + (j <= m.nx ÷ 2 + 1 ? j - 1 : j - 1 - m.nx)^2)) for i in 1:nk, j in 1:m.nx]
        wsym = ones(nk, m.nx); wsym[2:end, :] .= 2.0; iseven(m.nx) && (wsym[end, :] .= 1.0)
        KE1 = zeros(Float32, n); KE2 = zeros(Float32, n); APE = zeros(Float32, n); ub = zeros(Float32, m.nx, n)
        snap_t = unique(round.(Int, range(1, n; length=8))); snaps = zeros(Float32, m.nx, m.nx, length(snap_t))
        spec = zeros(maximum(kidx) + 1); nspec = 0
        for t in 1:n
            for _ in 1:sub
                qh = qg2_step(qh, h, m)
                t <= nly && (yq = qg2_step(yq, h, m))
            end
            ph = qg2_invert(qh, m)
            traj[:, t] .= qg2_observe(ph, m)
            e = qg2_energy(ph, m); KE1[t] = e.KE1; KE2[t] = e.KE2; APE[t] = e.APE
            ub[:, t] .= qg2_ubar(ph, m)[:, 1]
            (j = findfirst(==(t), snap_t)) === nothing || (snaps[:, :, j] .= m.pirf * qh[:, :, 1])
            if t % 10 == 0
                E1 = 0.5 .* wsym .* m.K2 .* abs2.(view(ph, :, :, 1)) ./ m.nx^4
                for I in eachindex(E1)
                    spec[kidx[I] + 1] += E1[I]
                end
                nspec += 1
            end
            if t <= nly
                d = nrm(yq, qh); acc += log(d / dq); yq = qh .+ (yq .- qh) .* (dq / d)
            end
        end
        extra = (KE1=KE1, KE2=KE2, APE=APE, ubar1=ub, snaps_q1=snaps, snap_t=snap_t, ke_spectrum=spec ./ max(nspec, 1))
    elseif name == "cdv"
        c = cdv_coeffs(p)
        x = [c.x1s, 0, 0, c.x4s, 0, 0] .+ 0.01 .* randn(rng, D)
        bufs = [zeros(D) for _ in 1:5]; bufy = [zeros(D) for _ in 1:5]
        for _ in 1:(ntr * sub)
            cdv_rk4!(x, h, c, bufs...)
        end
        y = x .+ d0 .* normalize(randn(rng, D))
        for t in 1:n
            for _ in 1:sub
                cdv_rk4!(x, h, c, bufs...)
                t <= nly && cdv_rk4!(y, h, c, bufy...)
            end
            traj[:, t] .= x
            if t <= nly
                d = norm(y .- x); acc += log(d / d0); y .= x .+ (d0 / d) .* (y .- x)
            end
        end
    elseif name == "cml"
        r, ε = p["r"], p["eps"]
        x = rand(rng, D)
        for _ in 1:ntr
            x = cml_step(x, r, ε)
        end
        y = x .+ d0 .* normalize(randn(rng, D))
        for t in 1:n
            x = cml_step(x, r, ε)
            traj[:, t] .= x
            if t <= nly
                y = cml_step(y, r, ε)
                d = norm(y .- x); acc += log(d / d0); y .= x .+ (d0 / d) .* (y .- x)
            end
        end
        turb = mean(<=(1.0), traj[:, (end - 10_000 + 1):end])
        turb > 0.05 || error(@sprintf("cml r=%g eps=%g: the truth laminarises (turbulent fraction %.3f) -- raise eps", r, ε, turb))
    else
        error("unknown system $name")
    end
    all(isfinite, traj) || error("integration blew up -- reduce int_dt")
    return traj, collect((1:n) .* dt), acc / (nly * dt), extra
end

"Chaté–Manneville map: tent of slope r on [0, 1], identity above 1 (laminar); diffusive coupling ε on a ring."
cml_f(x, r) = x <= 1.0 ? r * (0.5 - abs(x - 0.5)) : x
function cml_step(x::AbstractVector, r::Real, ε::Real)
    fx = cml_f.(x, r)
    return (1 - ε) .* fx .+ (ε / 2) .* (circshift(fx, 1) .+ circshift(fx, -1))
end

"""
캐시된 데이터 (없으면 생성; Duffing은 기존 파일, JetLat은 tools/fetch_jetlat.jl이 받은 파일). 반환: (path, traj, ts, lyap);
system.reduce ≠ "none"이면 축소 좌표 데이터 (path = 축소 파일) + `lift` (축소 계수 → 원래 상태, 둘 다 물리 좌표) + `base_path`.
"""
function system_data(cfg; say=println)
    b = _system_data_base(cfg; say=say)
    return get(cfg["system"], "reduce", "none") == "none" ? b : reduced_system_data(cfg, b; say=say)
end

function _system_data_base(cfg; say=println)
    path = data_path(cfg)
    if cfg["system"]["name"] == "jetlat"
        isfile(path) || error("JetLat data not found: $path -- run `julia --project=. tools/fetch_jetlat.jl` first")
        d = JLD2.load(path)
        return (path=path, traj=Matrix{Float64}(d["traj"]), ts=Vector{Float64}(d["ts"]), lyap=Float64(d["lyap"]), season=d["season"])
    end
    if cfg["system"]["name"] == "file"
        d = JLD2.load(path)
        size(d["traj"], 1) == cfg["system"]["N"] || error("file system: [system] N = $(cfg["system"]["N"]) ≠ data rows $(size(d["traj"], 1))")
        return (path=path, traj=Matrix{Float64}(d["traj"]), ts=Vector{Float64}(d["ts"]), lyap=Float64(d["lyap"]))
    end
    if cfg["system"]["name"] == "duffing"
        p = cfg["system"]
        @assert (p["alpha"], p["beta"], p["delta"], p["gamma"], p["omega"], p["dt"]) ==
                (DUFFING_PARAMS.α, DUFFING_PARAMS.β, DUFFING_PARAMS.δ, DUFFING_PARAMS.γ, DUFFING_PARAMS.ω, DUFFING_PARAMS.dt) "Duffing [system] parameters must match duffing_signature.jl"
        isfile(path) || error("Duffing data file not found: $path")
        d = JLD2.load(path)
        lyap = duffing_lyapunov()
        say(@sprintf("Duffing data %s (%d × %d), λ_max (true equations, Benettin) = %.4f", path, size(d["traj"])..., lyap))
        return (path=path, traj=Matrix{Float64}(d["traj"]), ts=Vector{Float64}(d["ts"]), lyap=lyap)
    end
    fpath = full_data_path(cfg)
    if !isfile(fpath)
        path = fpath
        say("generating $(cfg["system"]["name"]) data → $path")
        t0 = time()
        traj, ts, lyap, extra = generate_system_data(cfg)
        mkpath(dirname(path))
        tmp = path * ".tmp"
        jldsave(tmp; traj=traj, ts=ts, lyap=lyap, params=cfg["system"], extra...)
        mv(tmp, path; force=true)
        say(@sprintf("  generated %d × %d in %.1fs, λ_max = %.4f", size(traj)..., time() - t0, lyap))
    end
    path = data_path(cfg)
    if !isfile(path)                    # 관측 격자 파일 (λ_max는 적분 격자 계의 값 그대로)
        d = JLD2.load(fpath); k = obs_stride(cfg)
        tmp = path * ".tmp"
        jldsave(tmp; traj=d["traj"][1:k:end, :], ts=d["ts"], lyap=d["lyap"], params=cfg["system"])
        mv(tmp, path; force=true)
        say("observed every $k-th site of $(cfg["system"]["N"]) → $(obs_dim(cfg)) × $(size(d["traj"], 2)) ($path)")
    end
    d = JLD2.load(path)
    return (path=path, traj=d["traj"], ts=d["ts"], lyap=d["lyap"])
end

# ------------------------------------------------------------
# 상태 축소 (POD/EOF, Fourier): 축소 계수를 ESN 상태로 쓴다. 기준 상태(전체 격자 / 관측 좌표)는 lift로 복원해 signature,
# k–ω, 에너지 수지, 방정식 잔차(flow_map = 복원 → 적분 → 투영; 잔차에는 절단 오차가 들어간다)에 쓴다.
#   pod      training 창 [shift, shift + train_len)의 평균 m, 왼쪽 특이벡터 U[:, 1:k]: z = Uᵀ((x − m) ./ s), x ≈ m + s .* (U z).
#            s = training 표준편차 (단위가 섞인 계: maooam, cdv, jetlat, duffing) 또는 1 (같은 단위의 장: ks, lorenz96, cml, qg2, file —
#            표준화하면 작은 규모 / 약한 모드가 부풀려진다).
#   fourier  1차원 주기 격자 (L96, KS, CML, 공간 file): ĉ_j = rfft(x)_j / N, z = (Re ĉ0, Re ĉ1, Im ĉ1, …, Re ĉk, Im ĉk)
# ------------------------------------------------------------
const POD_STANDARDIZE = ("maooam", "cdv", "jetlat", "duffing")
function reduction_fit(cfg, X::AbstractMatrix)
    r = cfg["system"]["reduce"]; k = Int(cfg["system"]["reduce_k"]); D = size(X, 1)
    k >= 1 || error("system.reduce_k must be ≥ 1")
    if r == "pod"
        k <= D || error("reduce_k = $k > state dimension $D")
        cols = cfg["data"]["shift"]:(cfg["data"]["shift"] + cfg["data"]["train_len"] - 1)
        m = vec(mean(view(X, :, cols); dims=2))
        sc = cfg["system"]["name"] in POD_STANDARDIZE ? vec(std(view(X, :, cols); dims=2)) .+ eps() : ones(D)
        F = svd((X[:, cols] .- m) ./ sc)
        return (kind="pod", mean=m, scale=sc, modes=Matrix(F.U[:, 1:k]), explained=sum(abs2, F.S[1:k]) / sum(abs2, F.S), N=D)
    elseif r == "fourier"
        nm = cfg["system"]["name"]
        (nm in ("lorenz96", "ks", "cml") || (nm == "file" && cfg["system"]["spatial"])) ||
            error("reduce = fourier needs a 1-D periodic grid (lorenz96, ks, cml, spatial file); $nm" *
                  (nm == "qg2" ? " is already Fourier -- lower system.obs_k instead" : " -- use reduce = pod"))
        k <= D ÷ 2 - 1 || error("reduce_k = $k ≥ N/2")
        return (kind="fourier", mean=zeros(D), scale=ones(D), modes=zeros(D, 0), explained=NaN, N=D)
    end
    error("unknown system.reduce = $r")
end
function reduction_project(b, k::Int)
    b.kind == "pod" && return X -> b.modes' * ((X .- b.mean) ./ b.scale)
    return X -> begin
        C = rfft(X, 1) ./ b.N
        vcat(real.(C[1:1, :]), reduce(vcat, [vcat(real.(C[j + 1:j + 1, :]), imag.(C[j + 1:j + 1, :])) for j in 1:k]))
    end
end
function reduction_lift_fn(b, k::Int)
    b.kind == "pod" && return Z -> b.mean .+ b.scale .* (b.modes * Z)
    return Z -> begin
        Zm = Z isa AbstractVector ? reshape(Z, :, 1) : Z
        C = zeros(ComplexF64, b.N ÷ 2 + 1, size(Zm, 2))
        C[1, :] .= Zm[1, :]
        for j in 1:k
            C[j + 1, :] .= complex.(Zm[2j, :], Zm[2j + 1, :])
        end
        Xr = irfft(C .* b.N, b.N, 1)
        Z isa AbstractVector ? vec(Xr) : Xr
    end
end
"축소 파일의 기저 → lift (축소 계수 → 기준 상태, 물리 좌표). 축소가 없으면 nothing."
function reduction_lift(cfg)
    get(cfg["system"], "reduce", "none") == "none" && return nothing
    d = JLD2.load(reduced_data_path(cfg))
    b = (kind=d["basis_kind"], mean=d["basis_mean"], scale=d["basis_scale"], modes=d["basis_modes"], N=d["basis_N"])
    return reduction_lift_fn(b, Int(cfg["system"]["reduce_k"]))
end
function reduced_system_data(cfg, base; say=println)
    rpath = reduced_data_path(cfg); k = Int(cfg["system"]["reduce_k"])
    if !isfile(rpath)
        b = reduction_fit(cfg, base.traj)
        Z = reduction_project(b, k)(base.traj)
        err = let L = reduction_lift_fn(b, k), cols = (cfg["data"]["shift"] + cfg["data"]["train_len"]):min(size(base.traj, 2), cfg["data"]["shift"] + cfg["data"]["train_len"] + 20_000)
            X = base.traj[:, cols]; norm(L(Z[:, cols]) .- X) / norm(X .- vec(mean(X; dims=2)))
        end
        tmp = rpath * ".tmp"
        jldsave(tmp; traj=Z, ts=base.ts, lyap=base.lyap, basis_kind=b.kind, basis_mean=b.mean, basis_scale=b.scale, basis_modes=b.modes, basis_N=b.N,
                explained=b.explained, recon_err=err, base_path=base.path, params=cfg["system"])
        mv(tmp, rpath; force=true)
        say(@sprintf("reduced %s → %s %d (%d values; training-window variance explained %.3f, test reconstruction error %.3f of the anomaly) → %s",
                     base.path, b.kind, k, size(Z, 1), b.explained, err, rpath))
    end
    d = JLD2.load(rpath)
    bb = (kind=d["basis_kind"], mean=d["basis_mean"], scale=d["basis_scale"], modes=d["basis_modes"], N=d["basis_N"])
    return (path=rpath, traj=d["traj"], ts=d["ts"], lyap=d["lyap"], base_path=base.path, lift=reduction_lift_fn(bb, k),
            explained=d["explained"], recon_err=d["recon_err"])
end

"λ_max(1/시간) → Lyapunov time (step)."
lyap_steps(cfg, lyap) = 1 / (lyap * cfg["system"]["dt"])
"τ (step): 설정값, 0이면 tau_lyap × Lyapunov time."
experiment_tau(cfg, lyap) = cfg["experiment"]["tau"] > 0 ? cfg["experiment"]["tau"] :
    max(10, round(Int, cfg["experiment"]["tau_lyap"] * lyap_steps(cfg, lyap)))

"""
건강도 판정 창 W (step): 설정값, 0이면 "8 Lyapunov time 이상이고 chunk(ckpt_every·τ)를 나누는 τ의 가장 작은 배수"
(Duffing W = 600 ≈ 7.8 LT를 기준으로 잡은 규칙; τ = 5 LT, ckpt_every = 10이면 W = 2τ ≈ 10 LT). W와 τ는 한쪽이 다른 쪽의 배수여야 한다 (segment 경계 break 배치 고정).
"""
function experiment_health_window(cfg, lyap, τ::Int; ckpt_every::Int=10)
    w = Int(get(cfg["experiment"], "health_window", 0))
    w > 0 && return w
    m0 = max(1, ceil(Int, 8 * lyap_steps(cfg, lyap) / τ))
    return τ * minimum(m for m in 1:ckpt_every if m >= m0 && ckpt_every % m == 0; init=ckpt_every)   # 블록이 chunk(ckpt_every·τ)를 나눠야 한다
end

"R0b L0 Mahalanobis에 쓰는 분산 비율 ([experiment] mahal_var_frac; 없거나 1.0 = 전체 공분산, 기존 동작). taxonomy.jl `mahal_whitener` 참고."
experiment_mahal_var_frac(cfg) = Float64(get(cfg["experiment"], "mahal_var_frac", 1.0))

"R0a build_resync_context가 받는 시스템 spec (NamedTuple)."
function sys_from_config(cfg, dpath::AbstractString)
    e = cfg["esn"]; d = cfg["data"]; s = cfg["system"]
    return (name=s["name"], data_path=dpath, shift=d["shift"], train_len=d["train_len"], noise=d["noise"], noise_seed=1234,
        res_size=e["res_size"], radius=e["radius"], sparsity=e["degree"] / e["res_size"], leak=e["leak"], ridge=e["ridge"],
        washout=e["washout"], input_scaling=e["input_scaling"], bias_scaling=e["bias_scaling"], readout=e["readout"],
        sparse=e["sparse"],
        state_dim=state_dim(cfg), spatial=is_spatial(cfg),
        sign_regime_dim=s["name"] == "duffing" ? 1 : nothing,
        regime_K=8, normalize=d["normalize"], generator=nothing,
        ens_name="$(s["name"])_$(system_hash(cfg))_$(string(hash((sort(collect(filter(kv -> first(kv) != "reference_len", d)); by=first),
            sort(collect(filter(kv -> !(kv == ("bias_scaling" => 0.0) || kv == ("readout" => "linear") || kv == ("sparse" => false)), e)); by=first))); base=16))")
    # reference_len은 앙상블과 무관; bias_scaling = 0 / readout = linear(추가 전 동작)도 hash에서 빼서 기존 앙상블 디렉터리 유지
end

"R0a build_resync_context와 같은 데이터 분할/정규화/noise (단일 reservoir 탐색용; truth 길이 = predict_len)."
function prepare_data(cfg, traj::AbstractMatrix, predict_len::Int)
    d = cfg["data"]
    shift, tl = d["shift"], d["train_len"]
    need = shift + tl + predict_len
    need <= size(traj, 2) || error("data too short: need $need steps (shift+train_len+rollout), have $(size(traj, 2)) -- raise system.n_saved")
    tin = traj[:, shift:(shift + tl - 1)]; ttg = traj[:, (shift + 1):(shift + tl)]
    truth = traj[:, (shift + tl + 1):(shift + tl + predict_len)]
    x0 = traj[:, shift + tl]
    mu, sd = d["normalize"] ? (vec(mean(ttg; dims=2)), vec(std(ttg; dims=2)) .+ eps()) : (zeros(size(traj, 1)), ones(size(traj, 1)))
    nz(X) = (Matrix{Float64}(X) .- mu) ./ sd
    train_input = nz(tin)
    TestUtils.add_noise!(train_input; noiselevel=d["noise"], rng=MersenneTwister(1234))
    return (train_input=train_input, train_target=nz(ttg), truth=nz(truth), x0=(Vector{Float64}(x0) .- mu) ./ sd, mu=mu, sd=sd)
end

esn_base_config(cfg) = (e = cfg["esn"]; D = state_dim(cfg);
    TestUtils.default_experiment_config(in_size=D, res_size=e["res_size"], sparsity=e["degree"] / e["res_size"], radius=e["radius"],
        leak_coefficient=e["leak"], out_size=D, scaling=e["input_scaling"], training_method=StandardRidge(e["ridge"]),
        bias_scaling=Float64(e["bias_scaling"]), state_modifiers=TestUtils.readout_modifiers(e["readout"]),
        solver=QRSolver(), washout=e["washout"]))

# ------------------------------------------------------------
# 2. 공통 장기 통계 + 시스템 signature (정규화 좌표 기준)
# ------------------------------------------------------------

"시간 평균 공간 파수 파워 (길이 N÷2+1); stride로 표본 시각을 줄인다."
function spatial_spectrum(X::AbstractMatrix; stride::Int=5)
    idx = 1:stride:size(X, 2)
    acc = zeros(size(X, 1) ÷ 2 + 1)
    n = 0
    for t in idx
        c = view(X, :, t)
        all(isfinite, c) || continue
        acc .+= abs2.(rfft(collect(c))) ./ size(X, 1)
        n += 1
    end
    return n == 0 ? fill(NaN, length(acc)) : acc ./ n
end

"차원 평균 시간 자기상관 (lag 0..maxlag)."
function mean_acf(X::AbstractMatrix, maxlag::Int)
    D, T = size(X)
    maxlag = min(maxlag, T - 2)
    out = zeros(maxlag + 1)
    for d in 1:D
        x = view(X, d, :) .- mean(view(X, d, :))
        v = sum(abs2, x)
        v > 0 || continue
        for l in 0:maxlag
            out[l + 1] += sum(view(x, 1:(T - l)) .* view(x, (1 + l):T)) / v
        end
    end
    return out ./ D
end

log_spectrum_distance(a, b) = (ok = (a .> 0) .& (b .> 0); any(ok) ? sqrt(mean(abs2, log10.(a[ok]) .- log10.(b[ok]))) : NaN)

"""
공간 패턴의 이동 속도 (sites / 시간; 음수 = 인덱스 감소 방향 = L96의 서진): 지배 파수 k0(시간 평균 공간
스펙트럼 최대, k ≥ 1)의 Fourier 위상 φ(t)를 unwrap해 c = −N/(2π k0) · mean(Δφ)/dt. 패턴 x(j − c t)에 대해 정확.
"""
function wave_speed(X::AbstractMatrix, dt::Real; k0::Union{Nothing,Int}=nothing)
    D, T = size(X)
    T > 2 || return NaN
    k = k0 === nothing ? argmax(spatial_spectrum(X)[2:end]) : k0
    acc = 0.0; n = 0
    prev = NaN
    for t in 1:T
        c = view(X, :, t)
        all(isfinite, c) || (prev = NaN; continue)
        φ = angle(rfft(collect(c))[k + 1])
        if isfinite(prev)
            acc += mod(φ - prev + π, 2π) - π; n += 1
        end
        prev = φ
    end
    return n == 0 ? NaN : -D / (2π * k) * (acc / n) / dt
end

"KS: 스냅샷마다 u의 (주기 경계) 극대 개수."
ks_cells(X::AbstractMatrix) = [count(i -> X[i, t] > X[mod1(i - 1, end), t] && X[i, t] >= X[mod1(i + 1, end), t], 1:size(X, 1))
                               for t in axes(X, 2) if all(isfinite, view(X, :, t))]

"KS: 주기 중앙차분 u_x (격자 간격 단위)."
ks_ux(X::AbstractMatrix) = vec([(X[mod1(i + 1, size(X, 1)), t] - X[mod1(i - 1, size(X, 1)), t]) / 2 for i in axes(X, 1), t in axes(X, 2)])

w1_1d(a, b; nq=200) = (isempty(a) || isempty(b)) ? NaN :
    (p = range(0.5 / nq, 1 - 0.5 / nq; length=nq); mean(abs.(quantile(a, p) .- quantile(b, p))))

"""
    system_signature(name, X, ref; dt) -> NamedTuple (스칼라 지표, 작을수록 truth와 비슷)

`ref` = reference_signature(name, truth_long; dt). X는 정규화 좌표 궤적 (D × T); 비유한이면 NaN.
"""
function system_signature(name::AbstractString, X::AbstractMatrix, ref; dt::Real)
    name in ("file", "cml") && return NamedTuple()
    if name == "qg2"
        all(isfinite, X) || return (ke_w1=NaN, ape_w1=NaN, jet_w1=NaN)
        q = qg2_series(ref.model, denorm(X[:, 1:5:end], ref.mu, ref.sd))
        return (ke_w1=w1_1d(q.KE, ref.KE), ape_w1=w1_1d(q.APE, ref.APE), jet_w1=w1_1d(q.jet, ref.jet))
    end
    if name == "jetlat"
        all(isfinite, X) || return (lat_w1=NaN, regime_js=NaN, dwell_w1=NaN)
        lat = denorm(X, ref.mu, ref.sd)[1, :]
        lab = jetlat_label.(lat, Ref(ref.thr)); dw = run_lengths(lab)
        return (lat_w1=w1_1d(lat, ref.lat), regime_js=hist_js([count(==(r), lab) for r in 1:3], ref.occ),
            dwell_w1=length(dw) >= 2 ? w1_1d(log10.(dw), ref.log_dwell) : NaN)
    end
    if name == "cdv"
        all(isfinite, X) || return (x1_w1=NaN, dwell_w1=NaN, plane_js=NaN)
        Y = denorm(X, ref.mu, ref.sd)
        dw = cdv_dwell_times(view(Y, 1, :), ref.thr)
        return (x1_w1=w1_1d(Y[1, :], ref.x1), dwell_w1=length(dw) >= 2 ? w1_1d(log10.(dw), ref.log_dwell) : NaN,
            plane_js=hist_js(cdv_plane_hist(Y, ref.plane), ref.plane.h))
    end
    if name == "maooam"                                        # 정규화 좌표 → 모델 좌표 (ref가 mu, sd, 모델을 들고 있다)
        all(isfinite, X) || return (atm_energy_w1=NaN, ocean_energy_w1=NaN, jet_w1=NaN)
        q = maooam_series(ref.model, denorm(X[:, 1:5:end], ref.mu, ref.sd))
        return (atm_energy_w1=w1_1d(q.Ea, ref.Ea), ocean_energy_w1=w1_1d(q.Eo, ref.Eo), jet_w1=w1_1d(q.jet, ref.jet))
    end
    if name == "duffing"                                       # 물리 좌표 (normalize = false), 외력 위상이 필요한 section은 제외
        all(isfinite, X) || return (switch_rate=NaN, dwell_w1=NaN, energy_w1=NaN)
        dw = dwell_times(view(X, 1, :))
        return (switch_rate=switch_rate(view(X, 1, :)), dwell_w1=length(dw) >= 2 ? w1_samples(log10.(dw), ref.log_dwell) : NaN,
            energy_w1=w1_1d(duffing_energy(X[:, 1:5:end]), ref.E))
    end
    all(isfinite, X) || return name == "lorenz96" ? (energy_w1=NaN, wave_speed=NaN, spec_dist=NaN) :
                                                     (spec_dist=NaN, cells_w1=NaN, ux_w1=NaN)
    S = spatial_spectrum(X)
    if name == "lorenz96"
        E = vec(mean(abs2, X; dims=1)) ./ 2
        return (energy_w1=w1_1d(E, ref.E), wave_speed=wave_speed(X, dt; k0=ref.k0), spec_dist=log_spectrum_distance(S, ref.S))
    else
        return (spec_dist=log_spectrum_distance(S, ref.S), cells_w1=w1_1d(Float64.(ks_cells(X[:, 1:5:end])), ref.cells),
            ux_w1=w1_1d(ks_ux(X[:, 1:5:end]), ref.ux))
    end
end

function reference_signature(name::AbstractString, R::AbstractMatrix; dt::Real, mu=nothing, sd=nothing, p=nothing)
    name in ("file", "cml") && return NamedTuple()
    if name == "qg2"
        m = qg2_from_cfg(p); q = qg2_series(m, denorm(R[:, 1:5:end], mu, sd))
        return (model=m, mu=mu, sd=sd, KE=q.KE, APE=q.APE, jet=q.jet)
    end
    if name == "jetlat"
        lat = denorm(R, mu, sd)[1, :]; thr = jetlat_thresholds(lat); lab = jetlat_label.(lat, Ref(thr))
        return (mu=mu, sd=sd, thr=thr, lat=lat, occ=Float64[count(==(r), lab) for r in 1:3], log_dwell=log10.(Float64.(run_lengths(lab))))
    end
    if name == "cdv"
        Y = denorm(R, mu, sd); thr = cdv_threshold(view(Y, 1, :)); pl = cdv_plane_setup(Y)
        return (mu=mu, sd=sd, thr=thr, x1=Y[1, :], log_dwell=log10.(Float64.(cdv_dwell_times(view(Y, 1, :), thr))),
            plane=merge(pl, (h=cdv_plane_hist(Y, pl),)))
    end
    if name == "maooam"
        m = maooam_from_cfg(p)
        q = maooam_series(m, denorm(R[:, 1:5:end], mu, sd))
        return (model=m, mu=mu, sd=sd, Ea=q.Ea, Eo=q.Eo, jet=q.jet)
    end
    name == "duffing" && return (log_dwell=log10.(Float64.(dwell_times(view(R, 1, :)))), E=duffing_energy(R[:, 1:5:end]))
    S = spatial_spectrum(R)
    if name == "lorenz96"
        k0 = argmax(S[2:end])
        return (S=S, E=vec(mean(abs2, R; dims=1)) ./ 2, k0=k0, wave_speed=wave_speed(R, dt; k0=k0))
    else
        return (S=S, cells=Float64.(ks_cells(R[:, 1:5:end])), ux=ks_ux(R[:, 1:5:end]))
    end
end

const SIGNATURE_LABELS = Dict(
    "lorenz96" => [(:energy_w1, "W1(energy ½·mean x²) vs truth"), (:wave_speed, "wave speed (sites / time; truth dashed)"),
                   (:spec_dist, "log wavenumber-spectrum distance")],
    "ks" => [(:spec_dist, "log wavenumber-spectrum distance"), (:cells_w1, "W1(# cells per snapshot) vs truth"),
             (:ux_w1, "W1(u_x distribution) vs truth")],
    "qg2" => [(:ke_w1, "W1(large-scale kinetic energy) vs truth"), (:ape_w1, "W1(large-scale APE) vs truth"),
              (:jet_w1, "W1(jet strength max ū1) vs truth")],
    "jetlat" => [(:lat_w1, "W1(jet latitude) vs truth"), (:regime_js, "S/C/N regime occupancy JS vs truth"),
                 (:dwell_w1, "W1(log10 regime dwell) vs truth")],
    "cdv" => [(:x1_w1, "W1(zonal-flow index x1) vs truth"), (:dwell_w1, "W1(log10 regime dwell time) vs truth"),
              (:plane_js, "x1–x4 plane occupation JS vs truth")],
    "maooam" => [(:atm_energy_w1, "W1(atmospheric energy E_a) vs truth"), (:ocean_energy_w1, "W1(ocean energy E_o) vs truth"),
                 (:jet_w1, "W1(zonal jet index ψa1) vs truth")],
    "duffing" => [(:switch_rate, "well switches / 1000 steps (truth dashed)"), (:dwell_w1, "W1(log10 dwell time) vs truth"),
                  (:energy_w1, "W1(energy H) vs truth"), (:section_js, "Poincaré-section JS vs truth")])

# ------------------------------------------------------------
# 3. 상태 분류 (시스템 무관 3종 + 잠김): blow-up / fixed point / locked / wandering
# ------------------------------------------------------------
#   blow_up     비유한 또는 max|x| > 5 × (training max|x|)
#   fixed_point 차원 평균 시간 표준편차 < 5% × training 표준편차 평균
#   locked      시간 PSD의 spectral entropy가 training 창들 최소값의 0.8배 미만 (주기/준주기 고착 -- Duffing의
#               well_locked에 해당하는 "한 패턴에 갇힘")
#   wandering   그 외 (움직이지만 잠기지 않음)

"training 창(길이 L)들의 통계로 분류 임계값 계산."
function state_thresholds(train::AbstractMatrix, L::Int)
    sd = mean(std(train; dims=2))
    H = [temporal_entropy(view(train, :, s:(s + L - 1))) for s in 1:max(1, L ÷ 2):(size(train, 2) - L + 1)]
    return (blow=5 * maximum(abs, train), fixed=0.05 * sd, entropy=0.8 * minimum(H))
end

"차원 평균 시간 PSD의 정규화 spectral entropy (0 = 단일 주파수, 1 = 백색)."
function temporal_entropy(X::AbstractMatrix)
    P = vec(mean(abs2.(rfft(X .- mean(X; dims=2), 2)); dims=1))[2:end]
    s = sum(P)
    s > 0 || return 0.0
    p = P ./ s
    return -sum(x -> x > 0 ? x * log(x) : 0.0, p) / log(length(p))
end

function classify_state(X::AbstractMatrix, th)
    all(isfinite, X) || return :blowup
    maximum(abs, X) > th.blow && return :blowup
    mean(std(X; dims=2)) < th.fixed && return :fixed_point
    temporal_entropy(X) < th.entropy && return :locked
    return :wandering
end
const GENERIC_CLASS_ORDER = [:wandering, :locked, :fixed_point, :blowup]

# ------------------------------------------------------------
# 3b. "이 dynamics가 맞는가" 점검 (playground + 앙상블 파이프라인 공용 정의)
#   (1) 방정식 잔차: 예측 궤적 위에서 진짜 방정식 한 step과의 차이
#   (2) 물리 항등식: 장시간 평균에서 정확히 성립해야 하는 에너지 수지 / 보존량
#   (3) truth null band: 같은 길이 truth 창들로 만든 기준 분포
#   모든 함수는 물리 좌표를 받는다 (정규화 좌표 X → denorm(X, mu, sd)).
# ------------------------------------------------------------

denorm(X::AbstractMatrix, mu::AbstractVector, sd::AbstractVector) = X .* sd .+ mu

"""
    flow_map(cfg) -> Φ(x, t)

진짜 방정식으로 물리 좌표 상태 x를 저장 간격 dt만큼 전진 (데이터 생성과 같은 적분기·int_dt). t = x의 물리 시각
(외력이 있는 Duffing만 쓴다; 자율계는 무시). L96은 RK4, KS는 ETDRK4 (Fourier 공간에서 sub step 후 격자로 복귀),
Duffing은 int_dt RK4 (데이터 파일의 적분기와 달라 truth 잔차가 0은 아니지만 한 step 변화 대비 무시할 크기).
"""
function flow_map(cfg)
    if get(cfg["system"], "reduce", "none") != "none"     # 축소 좌표: 복원 → 기준 계 적분 → 투영
        c0 = deepcopy_config(cfg); c0["system"]["reduce"] = "none"; c0["system"]["reduce_k"] = 0
        base = flow_map(c0); L = reduction_lift(cfg)
        d = JLD2.load(reduced_data_path(cfg)); k = Int(cfg["system"]["reduce_k"])
        P = reduction_project((kind=d["basis_kind"], mean=d["basis_mean"], scale=d["basis_scale"], modes=d["basis_modes"], N=d["basis_N"]), k)
        return (z, t=0.0) -> vec(P(reshape(base(L(Vector{Float64}(z)), t), :, 1)))
    end
    p = cfg["system"]; h = p["int_dt"]; sub = round(Int, p["dt"] / h); D = p["N"]
    if p["name"] in ("file", "jetlat")                    # 방정식 없음: 잔차·참 λ는 NaN
        return (x, t=0.0) -> fill(NaN, length(x))
    elseif p["name"] == "qg2"
        m = qg2_from_cfg(p)
        return (x, t=0.0) -> begin
            qh = qg2_pv(qg2_unobserve(Vector{Float64}(x), m), m)
            for _ in 1:sub
                qh = qg2_step(qh, h, m)
            end
            qg2_observe(qg2_invert(qh, m), m)
        end
    elseif p["name"] == "duffing"
        return (x, t=0.0) -> begin
            y = Vector{Float64}(x); τ = Float64(t)
            for _ in 1:sub
                y = duffing_rk4(y, τ, h, p); τ += h
            end
            y
        end
    elseif p["name"] == "cml"
        r, ε = p["r"], p["eps"]
        return (x, t=0.0) -> cml_step(Vector{Float64}(x), r, ε)
    elseif p["name"] == "cdv"
        cc = cdv_coeffs(p)
        return (x, t=0.0) -> begin
            y = Vector{Float64}(x); bufs = [zeros(D) for _ in 1:5]
            for _ in 1:sub
                cdv_rk4!(y, h, cc, bufs...)
            end
            y
        end
    elseif p["name"] == "maooam"
        m = maooam_from_cfg(p)
        return (x, t=0.0) -> begin
            y = Vector{Float64}(x); bufs = [zeros(D) for _ in 1:5]
            for _ in 1:sub
                maooam_rk4!(y, h, m, bufs...)
            end
            y
        end
    elseif p["name"] == "lorenz96"
        F = p["F"]
        return (x, t=0.0) -> begin
            y = Vector{Float64}(x); bufs = [zeros(D) for _ in 1:5]
            for _ in 1:sub
                l96_rk4!(y, h, F, bufs...)
            end
            y
        end
    else
        c = ks_etdrk4_setup(D, p["L"], p["nu"], h; damp=get(p, "damp", 0.0))
        k = obs_stride(cfg); M = D ÷ k
        # 관측 격자(M)이면 Fourier 보간(영 채움)으로 적분 격자에 올려 적분하고 다시 M 지점만 읽는다. M 격자의 Nyquist
        # 모드는 버린다 -- 그래서 truth 위 잔차도 0이 아니고 보간 오차 크기다 (L=22, M=16: 에너지의 ~1e-4).
        lift(x) = k == 1 ? ks_fwd(x) : (u = rfft(x); w = zeros(ComplexF64, D ÷ 2 + 1); w[1:(M ÷ 2)] .= u[1:(M ÷ 2)] .* k; w)
        return (x, t=0.0) -> begin
            v = lift(Vector{Float64}(x))
            for _ in 1:sub
                v = ks_step(v, c)
            end
            ks_inv(v, c)[1:k:end]
        end
    end
end

"truth의 한 step 변화 크기 sqrt(mean_t ‖y_{t+1} − y_t‖²) — 방정식 잔차의 단위 (물리 좌표 Y)."
one_step_scale(Y::AbstractMatrix) = sqrt(mean(t -> sum(abs2, view(Y, :, t + 1) .- view(Y, :, t)), 1:(size(Y, 2) - 1)))

"""
    equation_residual(Φ, X, idx; scale) -> Vector

e_t = ‖Φ(x_t, time_of(t)) − x_{t+1}‖ / scale, t ∈ idx (X = 물리 좌표 예측; time_of = 열 → 물리 시각, 외력계만). 0 = 진짜 방정식과 같은 한 step,
1 ≈ truth의 한 step 변화만큼 틀림. 비유한 상태는 NaN.
"""
function equation_residual(Φ, X::AbstractMatrix, idx; scale::Real, time_of=nothing)
    r = fill(NaN, length(idx))
    for (j, t) in enumerate(idx)
        x = view(X, :, t); y = view(X, :, t + 1)
        (all(isfinite, x) && all(isfinite, y)) || continue
        r[j] = norm(Φ(x, time_of === nothing ? 0.0 : time_of(t)) .- y) / scale
    end
    return r
end

"""
    budget_ratio(name, Xphys, p) -> Float64 (장시간 평균에서 정확히 1)

L96: ⟨Σ_k x_k²⟩ / (F ⟨Σ_k x_k⟩) — 비선형 항이 에너지를 보존하므로 d/dt ½Σx² = −Σx² + FΣx 의 시간 평균.
KS:  ⟨Σ_k k²|û|²⟩ / (ν ⟨Σ_k k⁴|û|²⟩) — ½ d/dt ∫u² = ∫u_x² − ν∫u_xx² (비선형 항 기여 0). Nyquist 파수는 적분기와
     같이 0으로 둔다. 짧은 창에서는 요동하므로 truth 창들의 분포(null band)와 비교한다.
"""
function budget_ratio(name::AbstractString, X::AbstractMatrix, p)
    name in ("duffing", "maooam", "cdv", "qg2", "jetlat", "file") && return NaN          # 강제 감쇠계: 정확한 장시간 항등식 없음 (MAOOAM: 복사·마찰 수지가 상태 의존)
    all(isfinite, X) || return NaN
    name == "lorenz96" && return mean(abs2, X) / (p["F"] * mean(X))
    N = size(X, 1)
    k = (2π / p["L"]) .* collect(0:(N ÷ 2)); iseven(N) && (k[end] = 0.0)
    w = fill(2.0, length(k)); w[1] = 1.0; iseven(N) && (w[end] = 1.0)     # rfft 대칭 가중
    P = vec(sum(abs2.(rfft(X, 1)); dims=2))
    return sum(w .* k .^ 2 .* P) / (p["nu"] * sum(w .* k .^ 4 .* P))
end

"""
    ks_mean_drift(Xphys, c0, σu) -> Float64

KS의 공간 평균 ū(t)는 정확히 보존된다 (truth: 상수 c0). rms_t(ū(t) − c0)를 σu/√N (보존을 전혀 모르는
iid 격자장의 공간 평균 요동 크기) 단위로: 0 = 완전 보존, ~1 = 보존을 모름.
주의: 선형 ridge readout ESN에는 판별력이 없다 — 정규화된 training target이 Σ_i sd_i z_i = 0을 정확히 만족하고
readout W = Y Xᵀ(XXᵀ+λI)⁻¹은 target의 정확한 선형 불변량을 그대로 물려받아 출력이 항상 ~1e-15로 보존한다.
비선형 readout이나 target noise가 있을 때만 의미가 있다.
"""
function ks_mean_drift(X::AbstractMatrix, c0::Real, σu::Real)
    all(isfinite, X) || return NaN
    return sqrt(mean(abs2, vec(mean(X; dims=1)) .- c0)) / (σu / sqrt(size(X, 1)))
end

"""
    truth_windows(segs, W; n=20) -> Vector of views

truth 구간들(segs)에서 길이 W 창을 반씩 겹쳐 후보로 만들고, 그중 n개를 고르게 뽑는다 (결정적). null band용.
"""
function truth_windows(segs, W::Int; n::Int=20)
    cand = [(i, s) for (i, S) in enumerate(segs) for s in 1:max(1, W ÷ 2):(size(S, 2) - W + 1)]
    isempty(cand) && return SubArray[]
    pick = unique(round.(Int, range(1, length(cand); length=min(n, length(cand)))))
    return [view(segs[cand[j][1]], :, cand[j][2]:(cand[j][2] + W - 1)) for j in pick]
end

"""
null-band 판정 대상 지표: (key, 종류, 라벨). :dist = 작을수록 좋음 (truth 창들의 95% 분위 이하면 통과),
:range = truth 값 주변이어야 함 (truth 창들의 2.5–97.5% 분위 안이면 통과).
"""
const NULL_CHECKS = Dict(
    "lorenz96" => [(:marginal_w1, :dist, "marginal W1"), (:spec_dist, :dist, "wavenumber-spectrum distance"),
                   (:acf_dist, :dist, "temporal-ACF distance"), (:energy_w1, :dist, "energy W1"),
                   (:wave_speed, :range, "wave speed (sites / time)"), (:budget_ratio, :range, "energy budget ⟨Σx²⟩ / F⟨Σx⟩")],
    "ks" => [(:marginal_w1, :dist, "marginal W1"), (:spec_dist, :dist, "wavenumber-spectrum distance"),
             (:acf_dist, :dist, "temporal-ACF distance"), (:cells_w1, :dist, "# cells W1"), (:ux_w1, :dist, "u_x W1"),
             (:budget_ratio, :range, "energy budget ⟨u_x²⟩ / ν⟨u_xx²⟩")],
    "duffing" => [(:marginal_w1, :dist, "marginal W1"), (:acf_dist, :dist, "temporal-ACF distance"),
                  (:switch_rate, :range, "well switches / 1000 steps"), (:dwell_w1, :dist, "W1(log10 dwell time)"),
                  (:energy_w1, :dist, "energy W1")],
    "cml" => [(:marginal_w1, :dist, "marginal W1"), (:acf_dist, :dist, "temporal-ACF distance")],   # screening only so far
    "qg2" => [(:marginal_w1, :dist, "marginal W1"), (:acf_dist, :dist, "temporal-ACF distance"), (:ke_w1, :dist, "large-scale KE W1"),
              (:ape_w1, :dist, "large-scale APE W1"), (:jet_w1, :dist, "jet strength W1")],
    "jetlat" => [(:marginal_w1, :dist, "marginal W1"), (:acf_dist, :dist, "temporal-ACF distance"), (:lat_w1, :dist, "jet latitude W1"),
                 (:regime_js, :dist, "regime occupancy JS"), (:dwell_w1, :dist, "W1(log10 regime dwell)")],
    "cdv" => [(:marginal_w1, :dist, "marginal W1"), (:acf_dist, :dist, "temporal-ACF distance"), (:x1_w1, :dist, "zonal index x1 W1"),
              (:dwell_w1, :dist, "W1(log10 regime dwell)"), (:plane_js, :dist, "x1–x4 plane JS")],
    "maooam" => [(:marginal_w1, :dist, "marginal W1"), (:acf_dist, :dist, "temporal-ACF distance"),
                 (:atm_energy_w1, :dist, "atmospheric energy W1"), (:ocean_energy_w1, :dist, "ocean energy W1"),
                 (:jet_w1, :dist, "zonal jet index W1")],
    "file" => [(:marginal_w1, :dist, "marginal W1"), (:acf_dist, :dist, "temporal-ACF distance")])

for k in SPATIAL_SYSTEMS
    push!(NULL_CHECKS[k], (:kw_dist, :dist, "k–ω (space–time) spectrum distance"))
end

"mean_acf와 같은 정의를 FFT로 (O(D T log T)): 차원별 평균 제거 후 zero-padding 자기상관 / 분산, 차원 평균."
function mean_acf_fft(X::AbstractMatrix, maxlag::Int)
    D, T = size(X)
    maxlag = min(maxlag, T - 2)
    Z = X .- mean(X; dims=2)
    v = vec(sum(abs2, Z; dims=2))
    n = nextprod([2, 3, 5], 2T)
    F = rfft(hcat(Z, zeros(D, n - T)), 2)
    A = irfft(abs2.(F), n, 2)
    ok = v .> 0
    out = zeros(maxlag + 1)
    for d in 1:D
        ok[d] || continue
        out .+= view(A, d, 1:(maxlag + 1)) ./ v[d]
    end
    return out ./ D
end

"""
    kw_spectrum(X; nt) -> Matrix (D × nt÷2+1)

공간–시간(파수 k × 주파수 ω) power spectrum: 길이 nt의 반겹침 Hann 창마다 공간 fft(부호 있는 k; 전파 방향 보존)
× 시간 rfft, |.|² 평균. 비유한 창은 건너뛴다 (유한 창이 없으면 NaN).
부호 규약 (ω ≥ 0 반쪽만 저장): site 번호가 커지는 쪽으로 진행하는 파수 k의 파는 fft 행 index D − k + 1 (음의 k)에
나타난다 → 그림에서는 kw_signed_k(D)로 "진행 방향 부호 파수"(+ = site 증가 방향)로 바꿔 그린다.
"""
function kw_spectrum(X::AbstractMatrix; nt::Int)
    D, T = size(X)
    nt = min(nt, T)
    w = 0.5 .* (1 .- cos.(2π .* (0:(nt - 1)) ./ nt))
    acc = zeros(D, nt ÷ 2 + 1); n = 0
    for s in 1:max(1, nt ÷ 2):(T - nt + 1)
        Y = view(X, :, s:(s + nt - 1))
        all(isfinite, Y) || continue
        acc .+= abs2.(fft(rfft((Y .- mean(Y; dims=2)) .* w', 2), 1))
        n += 1
    end
    return n == 0 ? fill(NaN, D, nt ÷ 2 + 1) : acc ./ n
end

"kw_spectrum 행 순서 → 진행 방향 부호 파수 (+ = site 번호 증가 방향으로 진행). 반환 (정렬 순열, 파수)."
function kw_signed_k(D::Int)
    k = [-(r - 1 <= D ÷ 2 ? r - 1 : r - 1 - D) for r in 1:D]
    o = sortperm(k)
    return o, k[o]
end

"log10 k–ω spectrum RMS 차 (기준 power가 최대의 1e-4 이상인 칸만)."
function kw_distance(P::AbstractMatrix, Pref::AbstractMatrix)
    all(isfinite, P) || return NaN
    m = Pref .> 1e-4 * maximum(Pref)
    return sqrt(mean(abs2, log10.(P[m] .+ 1e-300) .- log10.(Pref[m])))
end

"""
    dyn_reference(cfg, Rlong, LT, mu, sd) -> NamedTuple

dynamics_metrics가 매번 다시 계산하지 않도록 긴 truth 기준(정규화 좌표 Rlong)의 통계를 미리 계산: marginal 분위,
ACF (앞 20000 step), signature 기준, k–ω spectrum (창 nt = 8 LT를 2의 거듭제곱으로, 64..1024).
"""
function dyn_reference(cfg, Rlong::AbstractMatrix, LT::Real, mu::AbstractVector, sd::AbstractVector; lift=nothing)
    name = cfg["system"]["name"]
    ml = round(Int, 5LT)
    nt = clamp(nextpow(2, round(Int, 8LT)), 64, 1024)
    P = range(0.5 / 200, 1 - 0.5 / 200; length=200)
    Rs = view(Rlong, :, 1:min(size(Rlong, 2), 20_000))
    # 축소 좌표면 marginal / ACF는 ESN 좌표에서, signature / k–ω / 에너지 수지는 복원한 기준 상태(물리 좌표, mu = 0, sd = 1)에서
    Rf = lift === nothing ? Rlong : lift(Rlong)
    smu, ssd = lift === nothing ? (mu, sd) : (zeros(size(Rf, 1)), ones(size(Rf, 1)))
    return (name=name, p=cfg["system"], dt=cfg["system"]["dt"], mu=mu, sd=sd, smu=smu, ssd=ssd, lift=lift, ml=ml, nt=nt, qp=P,
        rq=quantile(vec(Rlong), P), racf=mean_acf_fft(Rs, ml), sig=reference_signature(name, Rf; dt=cfg["system"]["dt"], mu=smu, sd=ssd, p=cfg["system"]),
        kw=kw_spectrum(Rf; nt=nt))
end

"""
    dynamics_metrics_kw(dr, X) -> (metrics NamedTuple, k–ω spectrum)

정규화 좌표 궤적 X의 장기 지표 한 묶음 (playground seed, truth null 창, 파이프라인 멤버 chunk 모두 같은 정의):
marginal_w1, acf_dist, signature (spec_dist, L96 energy_w1/wave_speed, KS cells_w1/ux_w1), budget_ratio, kw_dist.
"""
function dynamics_metrics_kw(dr, X::AbstractMatrix)
    fin = all(isfinite, X)
    Xf = (dr.lift === nothing || !fin) ? X : dr.lift(X)
    P = (fin && size(Xf, 2) >= dr.nt) ? kw_spectrum(Xf; nt=dr.nt) : fill(NaN, size(dr.kw))   # 창보다 짧은 조각은 NaN
    m = (marginal_w1=fin ? mean(abs.(quantile(vec(X), dr.qp) .- dr.rq)) : NaN,
        acf_dist=fin ? (a = mean_acf_fft(X, dr.ml); sqrt(mean(abs2, a .- dr.racf[1:length(a)]))) : NaN)
    sig = system_signature(dr.name, Xf, dr.sig; dt=dr.dt)
    return merge(m, sig, (budget_ratio=fin ? budget_ratio(dr.name, denorm(Xf, dr.smu, dr.ssd), dr.p) : NaN,
        kw_dist=kw_distance(P, dr.kw))), P
end
dynamics_metrics(dr, X::AbstractMatrix) = first(dynamics_metrics_kw(dr, X))

"""
    true_lyapunov(Φ, x0; n, transient, d0) -> λ per step

진짜 방정식의 Benettin 두-궤적 추정 (물리 좌표 x0에서 시작; 매 step 재정규화, 처음 transient step 제외).
외력계(Duffing)는 x0의 물리 시각 t0와 저장 간격 dt를 준다.
ESN 닫힌 루프 λ(esn_lyapunov_and_rollout, step 단위)와 같은 길이로 재서 추정 요동까지 맞춘 null band를 만든다.
"""
function true_lyapunov(Φ, x0::AbstractVector; n::Int, transient::Int, d0::Real=1e-8, rng=MersenneTwister(1), t0::Real=0.0, dt::Real=0.0)
    x = Vector{Float64}(x0)
    δ = randn(rng, length(x)); y = x .+ d0 .* δ ./ norm(δ)
    acc = 0.0; cnt = 0; τ = Float64(t0)
    for t in 1:n
        x = Φ(x, τ); y = Φ(y, τ); τ += dt
        d = norm(y .- x)
        (isfinite(d) && d > 0) || return NaN
        t > transient && (acc += log(d / d0); cnt += 1)
        y = x .+ (d0 / d) .* (y .- x)
    end
    return acc / max(cnt, 1)
end

"self-test: FFT ACF == 직접 ACF, k–ω spectrum이 진행파의 부호 있는 파수/주파수에 봉우리, truth 창의 budget ≈ 1은 호출부에서."
function self_test_dynamics_checks()
    rng = MersenneTwister(4)
    X = cumsum(randn(rng, 5, 400); dims=2)
    @assert maximum(abs.(mean_acf_fft(X, 30) .- mean_acf(X, 30))) < 1e-10 "mean_acf_fft != mean_acf"
    D, T = 16, 256
    W = [cos(2π * (3 * (j - 1) / D - 8 * (t - 1) / T)) for j in 1:D, t in 1:T]
    P = kw_spectrum(W; nt=T)
    ki, wi = Tuple(argmax(P))
    @assert (ki, wi) == (D - 3 + 1, 9) "kw_spectrum peak at $((ki, wi)), expected ($(D - 2), 9)"   # +site 방향 k=3 파 → 음의 k 행
    o, ks = kw_signed_k(D)
    @assert ks[findfirst(==(ki), o)] == 3 "kw_signed_k sign convention"
    return true
end

"null 값들 → 판정 구간 (lo, hi). 유한한 null 값이 없으면 (NaN, NaN)."
function null_interval(vals::AbstractVector, kind::Symbol)
    v = filter(isfinite, vals)
    isempty(v) && return (NaN, NaN)
    return kind === :dist ? (-Inf, quantile(v, 0.95)) : (quantile(v, 0.025), quantile(v, 0.975))
end
null_pass(x::Real, iv) = isfinite(x) && isfinite(iv[2]) && iv[1] <= x <= iv[2]

# ------------------------------------------------------------
# CDV 레짐 도구: x1 문턱 (truth 10%·90% 분위 중점 ± 폭의 10% 이력 띠), 체류 시간, x1–x4 평면 히스토그램 (25² + 밖 1칸)
# ------------------------------------------------------------
cdv_threshold(x1) = (a = quantile(x1, 0.1); b = quantile(x1, 0.9); c = (a + b) / 2; (lo=c - 0.1(b - a), hi=c + 0.1(b - a)))
"띠 흐름(x1 > hi) / 막힘(x1 < lo) 레짐 체류 시간 (step; 이력 띠 안은 직전 레짐 유지, 첫·끝 불완전 체류 제외)."
function cdv_dwell_times(x1::AbstractVector, thr)
    st = 0; start = 0; first_done = false; out = Int[]
    for (t, v) in enumerate(x1)
        new = v > thr.hi ? 1 : v < thr.lo ? -1 : st
        if new != st
            st != 0 && (first_done ? push!(out, t - start) : (first_done = true))
            st = new; start = t
        end
    end
    return out
end
function cdv_plane_setup(Y::AbstractMatrix; nb::Int=25)
    lo = [minimum(view(Y, d, :)) for d in (1, 4)]; hi = [maximum(view(Y, d, :)) for d in (1, 4)]; pad = 0.08 .* (hi .- lo)
    return (lo=lo .- pad, hi=hi .+ pad, nb=nb)
end
function cdv_plane_bin(pl, x1, x4)
    (pl.lo[1] <= x1 <= pl.hi[1] && pl.lo[2] <= x4 <= pl.hi[2]) || return pl.nb^2 + 1
    i = clamp(floor(Int, (x1 - pl.lo[1]) / (pl.hi[1] - pl.lo[1]) * pl.nb) + 1, 1, pl.nb)
    j = clamp(floor(Int, (x4 - pl.lo[2]) / (pl.hi[2] - pl.lo[2]) * pl.nb) + 1, 1, pl.nb)
    return (j - 1) * pl.nb + i
end
cdv_plane_hist(Y::AbstractMatrix, pl) = (h = zeros(pl.nb^2 + 1); for t in axes(Y, 2); h[cdv_plane_bin(pl, Y[1, t], Y[4, t])] += 1; end; h)
"""
JetLat 레짐 (남 / 중앙 / 북 = 1 / 2 / 3): truth jet 위도의 가우시안 KDE (폭 2°; 재분석 격자 때문에 1° 히스토그램은 들쭉날쭉하다)에서
35–65°N 사이의 깊은 국소 최소 두 개 (5° 이상 떨어진)를 경계로 (Woollings et al. 2010의 3봉 분포); 못 찾으면 3분위.
"""
function jetlat_thresholds(lat::AbstractVector; bw::Real=2.0)
    g = collect(range(floor(minimum(lat)), ceil(maximum(lat)); step=0.25))
    f = [sum(exp(-0.5 * ((x - v) / bw)^2) for v in lat) for x in g]
    mins = [i for i in 2:(length(f) - 1) if f[i] < f[i - 1] && f[i] <= f[i + 1] && 35 <= g[i] <= 65]
    sort!(mins; by=i -> f[i])
    pick = Int[]
    for i in mins
        all(abs(g[i] - g[j]) >= 5 for j in pick) && push!(pick, i)
        length(pick) == 2 && break
    end
    length(pick) == 2 || return (lo=quantile(lat, 1 / 3), hi=quantile(lat, 2 / 3))
    a, b = sort(g[pick])
    return (lo=a, hi=b)
end
jetlat_label(x, thr) = x < thr.lo ? 1 : x < thr.hi ? 2 : 3
"같은 값이 이어지는 길이 (첫·끝 불완전 구간 제외)."
function run_lengths(lab::AbstractVector)
    out = Int[]; start = 1
    for t in 2:length(lab)
        if lab[t] != lab[t - 1]
            start > 1 && push!(out, t - start)
            start = t
        end
    end
    return out
end

"두 히스토그램의 Jensen–Shannon divergence (자연로그)."
function hist_js(p, q)
    P = p ./ max(sum(p), 1); Q = q ./ max(sum(q), 1); M = (P .+ Q) ./ 2
    return sum(P[k] > 0 ? 0.5P[k] * log(P[k] / M[k]) : 0.0 for k in eachindex(P)) + sum(Q[k] > 0 ? 0.5Q[k] * log(Q[k] / M[k]) : 0.0 for k in eachindex(Q))
end

# ------------------------------------------------------------
# 4. 스트리밍 signature 누적기 (앙상블 파이프라인: 멤버 궤적 전체를 메모리에 두지 않고 chunk마다 갱신)
#    system_signature와 같은 정의를 고정 bin 히스토그램으로 근사 (W1 = bin CDF 차이 × 폭).
# ------------------------------------------------------------

"""
기준 궤적 R로 bin/파수 등 고정 설정을 만든다 (모든 멤버·truth가 같은 설정을 공유). Duffing은 Poincaré section 범위에
R의 물리 시각이 필요하다 (`t0` = R 첫 열의 시각).
"""
function signature_setup(name::AbstractString, R::AbstractMatrix; t0::Real=0.0, dt::Real=DUFFING_PARAMS.dt, mu=nothing, sd=nothing, p=nothing,
                         lift=nothing)
    lift === nothing && return merge(_signature_setup(name, R; t0=t0, dt=dt, mu=mu, sd=sd, p=p), (lift=nothing,))
    Rf = lift(R)                                              # 축소 좌표: 복원한 물리 좌표 상태에서 (mu = 0, sd = 1)
    return merge(_signature_setup(name, Rf; t0=t0, dt=dt, mu=zeros(size(Rf, 1)), sd=ones(size(Rf, 1)), p=p), (lift=lift,))
end
function _signature_setup(name::AbstractString, R::AbstractMatrix; t0::Real=0.0, dt::Real=DUFFING_PARAMS.dt, mu=nothing, sd=nothing, p=nothing)
    # External observations and CML have no system-specific signature defined.
    # In particular, X-only multiscale data must not inherit KS/Duffing diagnostics.
    name in ("file", "cml") && return (name=name, D=size(R, 1), nspec=0)
    if name == "qg2"
        m = qg2_from_cfg(p); q = qg2_series(m, denorm(R, mu, sd))
        qbins(v) = (lo = quantile(v, 0.0005); hi = quantile(v, 0.9995); w = hi - lo; range(lo - 0.25w, hi + 0.25w; length=121))
        return (name=name, D=size(R, 1), nspec=0, model=m, mu=mu, sd=sd, kbins=qbins(q.KE), abins=qbins(q.APE), jbins=qbins(q.jet))
    end
    if name == "jetlat"
        lat = denorm(R, mu, sd)[1, :]; thr = jetlat_thresholds(lat)
        dw = log10.(Float64.(run_lengths(jetlat_label.(lat, Ref(thr)))))
        return (name=name, D=size(R, 1), nspec=0, mu=mu, sd=sd, thr=thr, xbins=range(minimum(lat) - 10, maximum(lat) + 10; length=121),
            lbins=range(0.0, maximum(dw) + 0.5; length=121))
    end
    if name == "cdv"                                          # 모델 좌표의 x1 분포, 레짐 체류 시간, x1–x4 평면 점유
        Y = denorm(R, mu, sd); thr = cdv_threshold(view(Y, 1, :)); pl = cdv_plane_setup(Y)
        lo, hi = quantile(Y[1, :], 0.0005), quantile(Y[1, :], 0.9995); w = hi - lo
        dw = log10.(Float64.(cdv_dwell_times(view(Y, 1, :), thr)))
        return (name=name, D=size(R, 1), nspec=0, mu=mu, sd=sd, thr=thr, plane=pl, xbins=range(lo - 0.25w, hi + 0.25w; length=121),
            lbins=range(0.0, maximum(dw) + 0.5; length=121))
    end
    if name == "maooam"                                       # R = 정규화 좌표; 히스토그램은 모델 좌표의 E_a, E_o, jet
        m = maooam_from_cfg(p)
        q = maooam_series(m, denorm(R, mu, sd))
        bins(v) = (lo = quantile(v, 0.0005); hi = quantile(v, 0.9995); w = hi - lo; range(lo - 0.25w, hi + 0.25w; length=121))
        return (name=name, D=size(R, 1), nspec=0, model=m, mu=mu, sd=sd, ebins=bins(q.Ea), obins=bins(q.Eo), jbins=bins(q.jet))
    end
    if name == "duffing"
        E = duffing_energy(R[:, 1:5:end])
        dw = log10.(Float64.(dwell_times(view(R, 1, :))))
        ω = DUFFING_PARAMS.ω; hw = ω * dt / 2
        sec = [t for t in axes(R, 2) if abs(mod(ω * (t0 + (t - 1) * dt) + π, 2π) - π) <= hw]
        lo = vec(minimum(R[:, sec]; dims=2)); hi = vec(maximum(R[:, sec]; dims=2)); pad = 0.08 .* (hi .- lo)
        return (name=name, D=2, nspec=0, ebins=range(quantile(E, 0.0005) - 0.1, quantile(E, 0.9995) + 0.2; length=121),
            lbins=range(0.0, maximum(dw) + 0.5; length=121), slo=lo .- pad, shi=hi .+ pad, snb=25, ω=ω, hw=hw, dt=dt)
    end
    S = spatial_spectrum(R)
    if name == "lorenz96"
        E = vec(mean(abs2, R; dims=1)) ./ 2
        lo, hi = quantile(E, 0.0005), quantile(E, 0.9995)
        return (name=name, D=size(R, 1), k0=argmax(S[2:end]), nspec=length(S), ebins=range(0.5lo, 1.5hi; length=121))
    else
        ux = ks_ux(R[:, 1:5:end])
        a = quantile(abs.(ux), 0.9995)
        return (name=name, D=size(R, 1), nspec=length(S), cbins=-0.5:1:(3 * maximum(ks_cells(R[:, 1:5:end])) + 0.5), ubins=range(-2a, 2a; length=161))
    end
end

mutable struct SigAcc
    fin::Bool
    spec::Vector{Float64}; nspec::Int
    h1::Vector{Float64}; h2::Vector{Float64}             # L96: energy hist / —   KS: cells hist / u_x hist   Duffing: energy / log10 dwell   MAOOAM: E_a / E_o
    dphi::Float64; nphi::Int; last_phi::Float64; tcount::Int
    h3::Vector{Float64}                                   # Duffing: Poincaré-section grid (snb² 칸 + 밖 1칸)   MAOOAM: jet index ψa1
    state::Int; start::Int; first_done::Bool; nswitch::Int   # Duffing: hysteresis well 상태 (dwell_times / switch_rate와 같은 규칙)
end
SigAcc(ss) = ss.name in ("file", "cml") ? SigAcc(true, Float64[], 0, Float64[], Float64[], 0.0, 0, NaN, 0, Float64[], 0, 0, false, 0) :
    ss.name == "lorenz96" ? SigAcc(true, zeros(ss.nspec), 0, zeros(length(ss.ebins) + 1), Float64[], 0.0, 0, NaN, 0, Float64[], 0, 0, false, 0) :
    ss.name == "ks" ? SigAcc(true, zeros(ss.nspec), 0, zeros(length(ss.cbins) + 1), zeros(length(ss.ubins) + 1), 0.0, 0, NaN, 0, Float64[], 0, 0, false, 0) :
    ss.name == "qg2" ? SigAcc(true, Float64[], 0, zeros(length(ss.kbins) + 1), zeros(length(ss.abins) + 1), 0.0, 0, NaN, 0,
                              zeros(length(ss.jbins) + 1), 0, 0, false, 0) :
    ss.name == "jetlat" ? SigAcc(true, Float64[], 0, zeros(length(ss.xbins) + 1), zeros(length(ss.lbins) + 1), 0.0, 0, NaN, 0,
                                 zeros(3), 0, 0, false, 0) :
    ss.name == "cdv" ? SigAcc(true, Float64[], 0, zeros(length(ss.xbins) + 1), zeros(length(ss.lbins) + 1), 0.0, 0, NaN, 0,
                              zeros(ss.plane.nb^2 + 1), 0, 0, false, 0) :
    ss.name == "maooam" ? SigAcc(true, Float64[], 0, zeros(length(ss.ebins) + 1), zeros(length(ss.obins) + 1), 0.0, 0, NaN, 0,
                                 zeros(length(ss.jbins) + 1), 0, 0, false, 0) :
    SigAcc(true, Float64[], 0, zeros(length(ss.ebins) + 1), zeros(length(ss.lbins) + 1), 0.0, 0, NaN, 0, zeros(ss.snb^2 + 1), 0, 0, false, 0)

"값 v를 edges 기준 bin에 (양끝 밖은 첫/마지막 bin) 누적."
function hist_add!(h, edges, v)
    k = searchsortedlast(edges, v)
    h[clamp(k + 1, 1, length(h))] += 1
end

"X (D × T, 연속 시간 조각)로 누적 갱신. 비유한이면 이후 fin=false. Duffing은 X 첫 열의 물리 시각 `t0`가 필요하다 (section)."
function sig_update!(a::SigAcc, ss, X::AbstractMatrix; t0=nothing)
    a.fin || return a
    if !all(isfinite, X)
        a.fin = false
        return a
    end
    if ss.name in ("file", "cml")
        a.tcount += size(X, 2)
        return a
    end
    ss.lift === nothing || (X = ss.lift(X))                   # 축소 좌표 → 복원 (signature는 기준 상태에서)
    D, T = size(X)
    if ss.name == "qg2"
        q = qg2_series(ss.model, denorm(X, ss.mu, ss.sd))
        for t in 1:T
            hist_add!(a.h1, ss.kbins, q.KE[t]); hist_add!(a.h2, ss.abins, q.APE[t]); hist_add!(a.h3, ss.jbins, q.jet[t])
        end
        a.tcount += T
        return a
    end
    if ss.name == "jetlat"                                # 레짐 run 길이 (chunk 경계를 넘어 이어진다)
        lat = denorm(X, ss.mu, ss.sd)[1, :]
        for t in 1:T
            a.tcount += 1
            hist_add!(a.h1, ss.xbins, lat[t])
            r = jetlat_label(lat[t], ss.thr); a.h3[r] += 1
            if r != a.state
                a.state != 0 && (a.first_done ? hist_add!(a.h2, ss.lbins, log10(a.tcount - a.start)) : (a.first_done = true))
                a.state = r; a.start = a.tcount
            end
        end
        return a
    end
    if ss.name == "cdv"                                   # 레짐 상태 기계 = cdv_dwell_times와 같은 규칙 (chunk 경계를 넘어 이어진다)
        Y = denorm(X, ss.mu, ss.sd)
        for t in 1:T
            a.tcount += 1
            x1 = Y[1, t]
            hist_add!(a.h1, ss.xbins, x1)
            a.h3[cdv_plane_bin(ss.plane, x1, Y[4, t])] += 1
            new = x1 > ss.thr.hi ? 1 : x1 < ss.thr.lo ? -1 : a.state
            if new != a.state
                if a.state != 0
                    a.nswitch += 1
                    a.first_done ? hist_add!(a.h2, ss.lbins, log10(a.tcount - a.start)) : (a.first_done = true)
                end
                a.state = new; a.start = a.tcount
            end
        end
        return a
    end
    if ss.name == "maooam"
        q = maooam_series(ss.model, denorm(X, ss.mu, ss.sd))
        for t in 1:T
            hist_add!(a.h1, ss.ebins, q.Ea[t]); hist_add!(a.h2, ss.obins, q.Eo[t]); hist_add!(a.h3, ss.jbins, q.jet[t])
        end
        a.tcount += T
        return a
    end
    if ss.name == "duffing"
        θ = DUFFING_DWELL_THETA; nb = ss.snb
        for t in 1:T
            a.tcount += 1
            x1 = X[1, t]; x2 = X[2, t]
            hist_add!(a.h1, ss.ebins, x2^2 / 2 - DUFFING_PARAMS.α * x1^2 / 2 + DUFFING_PARAMS.β * x1^4 / 4)
            new = x1 > θ ? 1 : x1 < -θ ? -1 : a.state
            if new != a.state
                if a.state != 0
                    a.nswitch += 1
                    a.first_done ? hist_add!(a.h2, ss.lbins, log10(a.tcount - a.start)) : (a.first_done = true)
                end
                a.state = new; a.start = a.tcount
            end
            if t0 !== nothing && abs(mod(ss.ω * (t0 + (t - 1) * ss.dt) + π, 2π) - π) <= ss.hw
                if ss.slo[1] <= x1 <= ss.shi[1] && ss.slo[2] <= x2 <= ss.shi[2]
                    i = clamp(floor(Int, (x1 - ss.slo[1]) / (ss.shi[1] - ss.slo[1]) * nb) + 1, 1, nb)
                    j = clamp(floor(Int, (x2 - ss.slo[2]) / (ss.shi[2] - ss.slo[2]) * nb) + 1, 1, nb)
                    a.h3[(j - 1) * nb + i] += 1
                else
                    a.h3[end] += 1
                end
            end
        end
        return a
    end
    for t in 1:T
        a.tcount += 1
        c = view(X, :, t)
        if ss.name == "lorenz96"
            hist_add!(a.h1, ss.ebins, sum(abs2, c) / (2D))
            φ = angle(rfft(collect(c))[ss.k0 + 1])
            isfinite(a.last_phi) && (a.dphi += mod(φ - a.last_phi + π, 2π) - π; a.nphi += 1)
            a.last_phi = φ
        end
        if a.tcount % 5 == 1
            a.spec .+= abs2.(rfft(collect(c))) ./ D; a.nspec += 1
            if ss.name == "ks"
                hist_add!(a.h1, ss.cbins, count(i -> c[i] > c[mod1(i - 1, D)] && c[i] >= c[mod1(i + 1, D)], 1:D))
                for i in 1:D
                    hist_add!(a.h2, ss.ubins, (c[mod1(i + 1, D)] - c[mod1(i - 1, D)]) / 2)
                end
            end
        end
    end
    return a
end

"bin 히스토그램 두 개의 W1 (bin 중심 간격 = 폭; 양끝 overflow bin은 가장자리 bin으로 취급)."
function hist_w1(h1, h2, edges)
    (sum(h1) > 0 && sum(h2) > 0) || return NaN
    w = step(edges)
    return sum(abs.(cumsum(h1) ./ sum(h1) .- cumsum(h2) ./ sum(h2))) * w
end

"누적기 → system_signature와 같은 키의 NamedTuple (ref = 기준 궤적의 누적기)."
function sig_finalize(a::SigAcc, ref::SigAcc, ss; dt::Real)
    ss.name in ("file", "cml") && return NamedTuple()
    if ss.name == "qg2"
        a.fin || return (ke_w1=NaN, ape_w1=NaN, jet_w1=NaN)
        return (ke_w1=hist_w1(a.h1, ref.h1, ss.kbins), ape_w1=hist_w1(a.h2, ref.h2, ss.abins), jet_w1=hist_w1(a.h3, ref.h3, ss.jbins))
    end
    if ss.name == "jetlat"
        a.fin || return (lat_w1=NaN, regime_js=NaN, dwell_w1=NaN)
        return (lat_w1=hist_w1(a.h1, ref.h1, ss.xbins), regime_js=hist_js(a.h3, ref.h3),
            dwell_w1=sum(a.h2) >= 2 ? hist_w1(a.h2, ref.h2, ss.lbins) : NaN)
    end
    if ss.name == "cdv"
        a.fin || return (x1_w1=NaN, dwell_w1=NaN, plane_js=NaN)
        return (x1_w1=hist_w1(a.h1, ref.h1, ss.xbins), dwell_w1=sum(a.h2) >= 2 ? hist_w1(a.h2, ref.h2, ss.lbins) : NaN,
            plane_js=sum(a.h3) > 0 ? hist_js(a.h3, ref.h3) : NaN)
    end
    if ss.name == "maooam"
        a.fin || return (atm_energy_w1=NaN, ocean_energy_w1=NaN, jet_w1=NaN)
        return (atm_energy_w1=hist_w1(a.h1, ref.h1, ss.ebins), ocean_energy_w1=hist_w1(a.h2, ref.h2, ss.obins),
            jet_w1=hist_w1(a.h3, ref.h3, ss.jbins))
    end
    if ss.name == "duffing"
        a.fin || return (switch_rate=NaN, dwell_w1=NaN, energy_w1=NaN, section_js=NaN)
        js(p, q) = (P = p ./ max(sum(p), 1); Q = q ./ max(sum(q), 1); M = (P .+ Q) ./ 2;
            sum(P[k] > 0 ? 0.5P[k] * log(P[k] / M[k]) : 0.0 for k in eachindex(P)) + sum(Q[k] > 0 ? 0.5Q[k] * log(Q[k] / M[k]) : 0.0 for k in eachindex(Q)))
        return (switch_rate=1000a.nswitch / max(a.tcount, 1), dwell_w1=sum(a.h2) >= 2 ? hist_w1(a.h2, ref.h2, ss.lbins) : NaN,
            energy_w1=hist_w1(a.h1, ref.h1, ss.ebins), section_js=sum(a.h3) > 0 ? js(a.h3, ref.h3) : NaN)
    end
    if !a.fin
        return ss.name == "lorenz96" ? (energy_w1=NaN, wave_speed=NaN, spec_dist=NaN) : (spec_dist=NaN, cells_w1=NaN, ux_w1=NaN)
    end
    sd = log_spectrum_distance(a.spec ./ max(a.nspec, 1), ref.spec ./ max(ref.nspec, 1))
    if ss.name == "lorenz96"
        D = ss.D
        c = a.nphi == 0 ? NaN : -D / (2π * ss.k0) * (a.dphi / a.nphi) / dt
        return (energy_w1=hist_w1(a.h1, ref.h1, ss.ebins), wave_speed=c, spec_dist=sd)
    else
        return (spec_dist=sd, cells_w1=hist_w1(a.h1, ref.h1, ss.cbins), ux_w1=hist_w1(a.h2, ref.h2, ss.ubins))
    end
end
