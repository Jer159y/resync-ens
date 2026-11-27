# ============================================================
# maooam.jl — MAOOAM (Modular Arbitrary-Order Ocean-Atmosphere Model; De Cruz, Demaeyer & Vannitsem 2016, GMD 9, 2793)
#
# 2층 QG 대기 (순압 ψ_a, 경압 θ_a = 온도) + reduced-gravity QG 해양 (ψ_o) + 해양 온도 δT_o, β-평면 채널
# x ∈ [0, 2π/n), y ∈ [0, π] (길이 단위 L = scale/π, 시간 단위 1/f0). Galerkin 절단 후
#     dη_i/dt = Σ_{j,k} T_ijk η_j η_k   (η_0 = 1: 상수 / 선형 / 2차 항 하나의 텐서)
# 상태 순서 = Fortran과 같다: ψ_a(1:natm), θ_a(natm+1:2natm), ψ_o(2natm+1:2natm+noc), δT_o(2natm+noc+1:2natm+2noc).
#
# 텐서 계수는 Climdyn/MAOOAM fortran/aotensor_def.f90 그대로, 내적은 해석식(inprod_analytic.f90) 대신 기저 함수의
# Gauss–Legendre 수치 적분 (삼각함수 곱이라 40점이면 기계 정밀도). fortran/tests/test_aotensor.ref (기본 params.nml)와
# 대조: tests/maooam_check.jl.
#
# 기저 (⟨f, g⟩ = n/(2π²) ∫∫ f g dx dy 에 대해 정규직교):
#   대기  A(P)    = √2 cos(P y)
#         K(M, P) = 2 cos(M n x) sin(P y)
#         L(H, P) = 2 sin(H n x) sin(P y)
#   해양  φ(H, P) = 2 sin(H n x / 2) sin(P y)        (닫힌 해분: 경계에서 ψ_o = 0)
# 모드 선택 (modeselection.nml과 같은 규칙): 대기 블록 (M, P), M = 1..atm_x, P = 1..atm_y; M = 1이면 A, K, L 세 개, 아니면 K, L.
# 해양 (H, P), H = 1..oc_x, P = 1..oc_y. 표준 36변수 설정 (Vannitsem et al. 2015) = 대기 2x-2y (10) + 해양 2x-4y (8).
# ============================================================

"""
MAOOAM 물리 파라미터 세트 = Climdyn/MAOOAM fortran/params*.nml (원 논문들은 전이 구간 T_TRANS = 1e8 시간 단위):
  DDV2016  De Cruz, Demaeyer & Vannitsem 2016 (MAOOAM v1.0, GMD): 강한 결합 d = 1.1e-7 → 결합 저주파 변동 (LFV)
  VL2016   Vannitsem & Lucarini 2016 (J. Phys. A, covariant Lyapunov vectors): 약한 결합 d = 1e-8
  V2015    Vannitsem 2015 (GRL, 해양 혼합층 깊이와 NAO): DDV2016에서 h = 164.8
  NML      저장소 기본 fortran/params.nml — fortran/tests/test_aotensor.ref가 이 값으로 만들어졌다 (검증용)
"""
const MAOOAM_PARAM_SETS = let
    ddv = Dict{String,Any}(
        "scale" => 5e6, "f0" => 1.032e-4, "n" => 1.5, "rra" => 6370e3, "phi0_npi" => 0.25,
        "gp" => 3.1e-2, "r" => 1e-7, "h" => 136.5, "d" => 1.1e-7,
        "k" => 0.0145, "kp" => 0.0290, "sig0" => 0.1,
        "Go" => 5.46e8, "Co" => 310.0, "To0" => 301.46,
        "Ga" => 1e7, "Ca" => 103.3333, "epsa" => 0.7, "Ta0" => 289.30,
        "sc" => 1.0, "lambda" => 15.06, "RR" => 287.0, "sB" => 5.6e-8,
        "atm_x" => 2, "atm_y" => 2, "oc_x" => 2, "oc_y" => 4)
    vl = merge(ddv, Dict{String,Any}("h" => 500.0, "d" => 1e-8, "k" => 0.02, "kp" => 0.04, "Go" => 2e8, "Co" => 350.0, "To0" => 285.0,
        "Ca" => 87.5, "epsa" => 0.76, "Ta0" => 270.0, "lambda" => 20.0))
    Dict("DDV2016" => ddv, "VL2016" => vl, "V2015" => merge(ddv, Dict{String,Any}("h" => 164.8)),
        "NML" => merge(vl, Dict{String,Any}("r" => 1e-8, "Ca" => 100.0)))
end
const MAOOAM_PARAM_KEYS = sort(collect(keys(MAOOAM_PARAM_SETS["DDV2016"])))

"Golub–Welsch Gauss–Legendre 노드/가중치 ([−1, 1])."
function gauss_legendre(m::Int)
    β = [k / sqrt(4k^2 - 1) for k in 1:(m - 1)]
    E = eigen(SymTridiagonal(zeros(m), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

"모드 목록: 대기 (typ, M 또는 H, P), 해양 (H, P)."
function maooam_modes(atm_x::Int, atm_y::Int, oc_x::Int, oc_y::Int)
    atm = Tuple{Char,Int,Int}[]
    for M in 1:atm_x, P in 1:atm_y
        M == 1 && push!(atm, ('A', 0, P))
        push!(atm, ('K', M, P)); push!(atm, ('L', M, P))
    end
    # modeselection.nml 순서는 블록 (1,1), (1,2), (2,1), (2,2) … = M 바깥, P 안쪽 루프 — 위와 같다
    oc = [(H, P) for H in 1:oc_x for P in 1:oc_y]
    return atm, oc
end

"""
    maooam_model(p) -> NamedTuple

파라미터 Dict p (MAOOAM_PARAM_KEYS)로 텐서를 만든다. 반환: natm, noc, ndim, c0 (상수), Lin (선형, ndim × ndim),
quadratic 항 (qi, qj, qk, qv: dη_qi += qv·η_qj·η_qk), 기저 정보 (atm, oc, a = 대기 Laplacian 고유값, Mo = 해양),
G (= −L²/L_R²), 파생 파라미터 phys.
"""
function maooam_model(p::AbstractDict)
    n = Float64(p["n"]); f0 = Float64(p["f0"])
    L = p["scale"] / π
    LR = sqrt(p["gp"] * p["h"]) / f0
    phys = (n=n, L=L, G=-L^2 / LR^2, betp=L / p["rra"] * cos(p["phi0_npi"] * π) / sin(p["phi0_npi"] * π),
        rp=p["r"] / f0, dp=p["d"] / f0, kd=2p["k"], kdp=p["kp"], sig0=p["sig0"], sc=p["sc"],
        Cpo=p["Co"] / (p["Go"] * f0) * p["RR"] / (f0^2 * L^2), Lpo=p["lambda"] / (p["Go"] * f0),
        Cpa=p["Ca"] / (p["Ga"] * f0) * p["RR"] / (f0^2 * L^2) / 2, Lpa=p["lambda"] / (p["Ga"] * f0),
        sBpo=4 * p["sB"] * p["To0"]^3 / (p["Go"] * f0), sBpa=8 * p["epsa"] * p["sB"] * p["Ta0"]^3 / (p["Go"] * f0),
        LSBpo=2 * p["epsa"] * p["sB"] * p["To0"]^3 / (p["Ga"] * f0), LSBpa=8 * p["epsa"] * p["sB"] * p["Ta0"]^3 / (p["Ga"] * f0))
    atm, oc = maooam_modes(Int(p["atm_x"]), Int(p["atm_y"]), Int(p["oc_x"]), Int(p["oc_y"]))
    na, no = length(atm), length(oc)
    # --- 수치 내적: 기저 값과 x, y 미분을 Gauss 격자에서 ---
    xg, wx = gauss_legendre(48); yg, wy = gauss_legendre(48)
    X = (xg .+ 1) .* (π / n); Y = (yg .+ 1) .* (π / 2)                 # [0, 2π/n] × [0, π]
    W = vec([wx[i] * (π / n) * wy[j] * (π / 2) for i in eachindex(xg), j in eachindex(yg)]) .* (n / (2π^2))
    xs = vec([X[i] for i in eachindex(xg), j in eachindex(yg)]); ys = vec([Y[j] for i in eachindex(xg), j in eachindex(yg)])
    function afun(m)
        t, M, P = m
        t == 'A' && return (√2 .* cos.(P .* ys), zeros(length(xs)), -√2 * P .* sin.(P .* ys))
        t == 'K' && return (2 .* cos.(M * n .* xs) .* sin.(P .* ys), -2M * n .* sin.(M * n .* xs) .* sin.(P .* ys),
                            2P .* cos.(M * n .* xs) .* cos.(P .* ys))
        return (2 .* sin.(M * n .* xs) .* sin.(P .* ys), 2M * n .* cos.(M * n .* xs) .* sin.(P .* ys), 2P .* sin.(M * n .* xs) .* cos.(P .* ys))
    end
    ofun((H, P)) = (2 .* sin.(H * n / 2 .* xs) .* sin.(P .* ys), H * n .* cos.(H * n / 2 .* xs) .* sin.(P .* ys),
                    2P .* sin.(H * n / 2 .* xs) .* cos.(P .* ys))
    Fa = afun.(atm); Fo = ofun.(oc)
    ip(f, g) = sum(W .* f .* g)
    jac(f, g) = f[2] .* g[3] .- f[3] .* g[2]                          # J(f, g) = ∂x f ∂y g − ∂y f ∂x g
    a = [-(n^2) * (m[1] == 'A' ? 0 : m[2])^2 - m[3]^2 for m in atm]    # ∇²F_i = a_i F_i
    Mo = [-(n^2) * (H / 2)^2 - P^2 for (H, P) in oc]                   # ∇²φ_i = M_i φ_i
    c = [ip(Fa[i][1], Fa[j][2]) for i in 1:na, j in 1:na]             # ⟨F_i, ∂x F_j⟩
    g = [ip(Fa[i][1], jac(Fa[j], Fa[k])) for i in 1:na, j in 1:na, k in 1:na]
    b = [a[k] * g[i, j, k] for i in 1:na, j in 1:na, k in 1:na]        # ⟨F_i, J(F_j, ∇²F_k)⟩
    s = [ip(Fa[i][1], Fo[j][1]) for i in 1:na, j in 1:no]             # ⟨F_i, φ_j⟩
    d = [s[i, j] * Mo[j] for i in 1:na, j in 1:no]                    # ⟨F_i, ∇²φ_j⟩
    K = [s[j, i] * a[j] for i in 1:no, j in 1:na]                     # ⟨φ_i, ∇²F_j⟩
    Wo = [s[j, i] for i in 1:no, j in 1:na]                           # ⟨φ_i, F_j⟩
    No = [ip(Fo[i][1], Fo[j][2]) for i in 1:no, j in 1:no]            # ⟨φ_i, ∂x φ_j⟩
    O = [ip(Fo[i][1], jac(Fo[j], Fo[k])) for i in 1:no, j in 1:no, k in 1:no]
    Co = [Mo[k] * O[i, j, k] for i in 1:no, j in 1:no, k in 1:no]
    clean!(A) = (A[abs.(A) .< 1e-12] .= 0.0; A)
    foreach(clean!, (c, g, b, s, d, K, Wo, No, O, Co))
    # --- 텐서 조립 (aotensor_def.f90 compute_aotensor와 같은 항, 같은 부호) ---
    nd = 2na + 2no
    ψ(i) = i; θ(i) = i + na; A(i) = i + 2na; T(i) = i + 2na + no
    c0 = zeros(nd); Lin = zeros(nd, nd); Q = Dict{Tuple{Int,Int,Int},Float64}()
    δ(i, j) = i == j ? 1.0 : 0.0
    lin!(i, j, v) = (Lin[i, j] += v)
    quad!(i, j, k, v) = v == 0 || (key = (i, min(j, k), max(j, k)); Q[key] = get(Q, key, 0.0) + v)
    ph = phys
    c0[θ(1)] += ph.Cpa / (1 - a[1] * ph.sig0)
    for i in 1:na
        for j in 1:na
            lin!(ψ(i), ψ(j), -(c[i, j] * ph.betp) / a[i] - ph.kd * δ(i, j) / 2)
            lin!(θ(i), ψ(j), (a[i] * δ(i, j) * ph.kd * ph.sig0) / (-2 + 2a[i] * ph.sig0))
            lin!(ψ(i), θ(j), ph.kd * δ(i, j) / 2)
            lin!(θ(i), θ(j), (-(ph.sig0 * (2c[i, j] * ph.betp + a[i] * δ(i, j) * (ph.kd + 4ph.kdp))) +
                              2 * (ph.LSBpa + ph.sc * ph.Lpa) * δ(i, j)) / (-2 + 2a[i] * ph.sig0))
            for k in 1:na
                quad!(ψ(i), ψ(j), ψ(k), -b[i, j, k] / a[i])
                quad!(ψ(i), θ(j), θ(k), -b[i, j, k] / a[i])
                quad!(θ(i), ψ(j), θ(k), (g[i, j, k] - b[i, j, k] * ph.sig0) / (-1 + a[i] * ph.sig0))
                quad!(θ(i), θ(j), ψ(k), (b[i, j, k] * ph.sig0) / (1 - a[i] * ph.sig0))
            end
        end
        for j in 1:no
            lin!(ψ(i), A(j), ph.kd * d[i, j] / (2a[i]))
            lin!(θ(i), A(j), ph.kd * d[i, j] * ph.sig0 / (2 - 2a[i] * ph.sig0))
            lin!(θ(i), T(j), s[i, j] * (2ph.LSBpo + ph.Lpa) / (2 - 2a[i] * ph.sig0))
        end
    end
    for i in 1:no
        for j in 1:na
            lin!(A(i), ψ(j), K[i, j] * ph.dp / (Mo[i] + ph.G))
            lin!(A(i), θ(j), -K[i, j] * ph.dp / (Mo[i] + ph.G))
        end
        for j in 1:no
            lin!(A(i), A(j), -(No[i, j] * ph.betp + Mo[i] * (ph.rp + ph.dp) * δ(i, j)) / (Mo[i] + ph.G))
            for k in 1:no
                quad!(A(i), A(j), A(k), -Co[i, j, k] / (Mo[i] + ph.G))
            end
        end
    end
    for i in 1:no
        c0[T(i)] += ph.Cpo * Wo[i, 1]
        for j in 1:na
            lin!(T(i), θ(j), Wo[i, j] * (2ph.sc * ph.Lpo + ph.sBpa))
        end
        for j in 1:no
            lin!(T(i), T(j), -(ph.Lpo + ph.sBpo) * δ(i, j))
            for k in 1:no
                quad!(T(i), A(j), T(k), -O[i, j, k])
            end
        end
    end
    filter!(kv -> abs(last(kv)) >= 1e-14, Q)
    ks = sort(collect(keys(Q)))
    return (natm=na, noc=no, ndim=nd, c0=c0, Lin=Lin, qi=getindex.(ks, 1), qj=getindex.(ks, 2), qk=getindex.(ks, 3),
        qv=[Q[k] for k in ks], atm=atm, oc=oc, a=a, Mo=Mo, G=ph.G, phys=ph)
end

"dη/dt (in-place)."
function maooam_rhs!(du::AbstractVector, x::AbstractVector, m)
    mul!(du, m.Lin, x)
    du .+= m.c0
    @inbounds for q in eachindex(m.qv)
        du[m.qi[q]] += m.qv[q] * x[m.qj[q]] * x[m.qk[q]]
    end
    return du
end

function maooam_rk4!(x, h, m, k1, k2, k3, k4, tmp)
    maooam_rhs!(k1, x, m); tmp .= x .+ (h / 2) .* k1
    maooam_rhs!(k2, tmp, m); tmp .= x .+ (h / 2) .* k2
    maooam_rhs!(k3, tmp, m); tmp .= x .+ h .* k3
    maooam_rhs!(k4, tmp, m); x .+= (h / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

"상태 블록의 index 범위: (psi_a, theta_a, psi_o, T_o)."
maooam_blocks(m) = (psi_a=1:m.natm, theta_a=(m.natm + 1):(2m.natm), psi_o=(2m.natm + 1):(2m.natm + m.noc),
    T_o=(2m.natm + m.noc + 1):m.ndim)

"""
    maooam_fields(m, x; nx=48, ny=32) -> (xs, ys, ψ_a, θ_a, ψ_o, T_o)

스펙트럼 계수 x를 (nx × ny) 격자의 장으로 복원 (시각화: 대기 jet / 해양 double gyre / 해양 온도).
"""
function maooam_fields(m, x::AbstractVector; nx::Int=48, ny::Int=32)
    n = m.phys.n
    xs = range(0, 2π / n; length=nx); ys = range(0, π; length=ny)
    fa(t, M, P, X, Y) = t == 'A' ? √2 * cos(P * Y) : t == 'K' ? 2cos(M * n * X) * sin(P * Y) : 2sin(M * n * X) * sin(P * Y)
    fo(H, P, X, Y) = 2sin(H * n / 2 * X) * sin(P * Y)
    b = maooam_blocks(m)
    F(idx, f, modes) = [sum(x[idx[i]] * f(modes[i]..., X, Y) for i in eachindex(modes)) for X in xs, Y in ys]
    return xs, ys, F(b.psi_a, fa, m.atm), F(b.theta_a, fa, m.atm), F(b.psi_o, fo, m.oc), F(b.T_o, fo, m.oc)
end

"모드 라벨 (그림·로그용): ψa1 … θa10, ψo1 …, To8."
maooam_labels(m) = vcat(["ψa$i" for i in 1:m.natm], ["θa$i" for i in 1:m.natm], ["ψo$i" for i in 1:m.noc], ["To$i" for i in 1:m.noc])

const _MAOOAM_CACHE = Dict{UInt,Any}()
"[system] 설정 Dict → 모델 (같은 파라미터면 캐시 재사용)."
maooam_from_cfg(p::AbstractDict) = (key = hash([(k, p[k]) for k in MAOOAM_PARAM_KEYS]);
    get!(() -> maooam_model(p), _MAOOAM_CACHE, key))

"""
signature용 스칼라 계열 (무차원 모델 좌표 X, 열 = 시각):
  E_a  대기 총에너지 ½ Σ (−a_i) ψ_a,i² + (−a_i + 1/σ0) θ_a,i²   (운동 + 위치; 무강제·비점성에서 보존)
  E_o  해양 총에너지 ½ Σ (−M_i − G) ψ_o,i²                       (운동 + 자유 표면 위치)
  jet  ψ_a,1 = A(1) 모드 진폭 (띠 평균 순압 흐름 = 중위도 jet 세기 지수)
"""
function maooam_series(m, X::AbstractMatrix)
    b = maooam_blocks(m)
    wψ = -m.a; wθ = -m.a .+ 1 / m.phys.sig0; wo = -m.Mo .- m.G
    Ea = [0.5 * (sum(wψ .* view(X, b.psi_a, t) .^ 2) + sum(wθ .* view(X, b.theta_a, t) .^ 2)) for t in axes(X, 2)]
    Eo = [0.5 * sum(wo .* view(X, b.psi_o, t) .^ 2) for t in axes(X, 2)]
    return (Ea=Ea, Eo=Eo, jet=Vector{Float64}(X[1, :]))
end
