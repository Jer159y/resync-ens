# ============================================================
# qg2.jl — 2층 준지균(QG) 모델, 이중 주기 영역 (pyqg QGModel과 같은 식·기본값; Ross et al. 2023 eddy / jet 설정)
#
#   q1 = ∇²ψ1 + F1(ψ2 − ψ1),   q2 = ∇²ψ2 + F2(ψ1 − ψ2),   F1 = rd⁻²/(1 + δ), F2 = δ F1, δ = H1/H2
#   ∂t q_n + J(ψ_n, q_n) + U_n ∂x q_n + Q_y,n ∂x ψ_n = −δ_n2 r_ek ∇²ψ2     (+ 지수 스펙트럼 필터 = 작은 규모 소산)
#   Q_y1 = β + F1(U1 − U2),   Q_y2 = β − F2(U1 − U2)
# 배경 시어 U1 − U2가 경압 불안정으로 에디를 만들고 바닥 마찰 r_ek가 에너지를 뺀다.
#   eddy (pyqg 기본):  r_ek = 5.787e-7, δ = 0.25, β = 1.5e-11 → 변형 반경 규모의 등방 에디
#   jet:               r_ek = 7e-8,     δ = 0.1,  β = 1e-11   → 지속되는 띠 모양 jet (병합·이동)
# 공통: L = 1e6 m, rd = 15 km, H1 = 500 m, U1 = 0.025, U2 = 0, nx = 64, dt = 3600 s.
# pyqg와의 차이: 시간 적분이 AB3 대신 RK4 (필터는 pyqg처럼 매 step 뒤 q̂에 곱한다).
#
# 상태 (ESN이 보는 좌표): 각 층 ψ의 저파수 Fourier 계수 |k|, |l| ≤ K (index 단위, (0,0) 제외, k = 0 행은 l > 0만)의
# 실부·허부 → 층당 4K(K+1)개. 전체 해상도 장은 저장하지 않고, 진단(에너지, 띠 평균 u(y), 스냅샷)만 따로 저장한다.
# ============================================================

"pyqg eddy / jet 설정 (Ross et al. 2023, pyqg_parameterization_benchmarks)."
const QG2_PARAM_SETS = Dict(
    "eddy" => Dict{String,Any}("L" => 1e6, "nx" => 64, "beta" => 1.5e-11, "rd" => 15000.0, "delta" => 0.25, "H1" => 500.0,
        "U1" => 0.025, "U2" => 0.0, "rek" => 5.787e-7, "filterfac" => 23.6, "obs_k" => 4),
    "jet" => Dict{String,Any}("L" => 1e6, "nx" => 64, "beta" => 1e-11, "rd" => 15000.0, "delta" => 0.1, "H1" => 500.0,
        "U1" => 0.025, "U2" => 0.0, "rek" => 7e-8, "filterfac" => 23.6, "obs_k" => 4))

"모델 (격자, 파수, 역산 행렬, 필터, FFT plan, 관측 모드 목록)."
function qg2_model(p::AbstractDict)
    nx = Int(p["nx"]); L = Float64(p["L"]); dx = L / nx
    δ = Float64(p["delta"]); F1 = p["rd"]^-2 / (1 + δ); F2 = δ * F1
    U1, U2 = Float64(p["U1"]), Float64(p["U2"])
    Qy = (p["beta"] + F1 * (U1 - U2), p["beta"] - F2 * (U1 - U2))
    nk = nx ÷ 2 + 1
    ki = collect(0:(nx ÷ 2)); li = [j <= nx ÷ 2 ? j : j - nx for j in 0:(nx - 1)]   # index 파수 (x: rfft, y: fft)
    k = (2π / L) .* ki; l = (2π / L) .* li
    K2 = [k[i]^2 + l[j]^2 for i in 1:nk, j in 1:nx]
    det = K2 .* (K2 .+ F1 .+ F2)
    inv(d) = d == 0 ? 0.0 : 1 / d
    a11 = -(K2 .+ F2) .* inv.(det); a12 = -F1 .* inv.(det); a21 = -F2 .* inv.(det); a22 = -(K2 .+ F1) .* inv.(det)
    wvx = [sqrt((k[i] * dx)^2 + (l[j] * dx)^2) for i in 1:nk, j in 1:nx]
    filt = [w <= 0.65π ? 1.0 : exp(-p["filterfac"] * (w - 0.65π)^4) for w in wvx]
    K = Int(p["obs_k"])
    obs = [(i, j) for i in 0:K for j in -K:K if (i > 0 || j > 0)]          # (kx, ly) index 쌍
    oidx = [(i + 1, j >= 0 ? j + 1 : nx + j + 1) for (i, j) in obs]
    A = zeros(nx, nx)
    return (nx=nx, L=L, dx=dx, F1=F1, F2=F2, δ=δ, del1=δ / (1 + δ), del2=1 / (1 + δ), U=(U1, U2), Qy=Qy, rek=Float64(p["rek"]),
        ik=im .* k, il=im .* l, K2=K2, a=(a11, a12, a21, a22), filt=filt, obs=obs, oidx=oidx, nobs=2 * 2 * length(obs),
        prf=plan_rfft(A), pirf=plan_irfft(rfft(A), nx))
end

"q̂ (nk × nx × 2) → ψ̂."
function qg2_invert(qh, m)
    a11, a12, a21, a22 = m.a
    q1 = view(qh, :, :, 1); q2 = view(qh, :, :, 2)
    return cat(a11 .* q1 .+ a12 .* q2, a21 .* q1 .+ a22 .* q2; dims=3)
end
"ψ̂ → q̂."
qg2_pv(ph, m) = cat(-m.K2 .* ph[:, :, 1] .+ m.F1 .* (ph[:, :, 2] .- ph[:, :, 1]),
                    -m.K2 .* ph[:, :, 2] .+ m.F2 .* (ph[:, :, 1] .- ph[:, :, 2]); dims=3)

"dq̂/dt (pyqg kernel의 _do_advection + _do_friction과 같은 항)."
function qg2_tendency(qh, m)
    ph = qg2_invert(qh, m)
    out = similar(qh)
    ikv = reshape(m.ik, :, 1); ilv = reshape(m.il, 1, :)
    for n in 1:2
        P = view(ph, :, :, n)
        u = m.pirf * (-ilv .* P); v = m.pirf * (ikv .* P); q = m.pirf * view(qh, :, :, n)
        uqh = m.prf * ((u .+ m.U[n]) .* q); vqh = m.prf * (v .* q)
        out[:, :, n] .= -(ikv .* uqh .+ ilv .* vqh .+ (m.Qy[n] .* ikv) .* P)
    end
    out[:, :, 2] .+= m.rek .* m.K2 .* view(ph, :, :, 2)
    return out
end

"RK4 한 step + 필터."
function qg2_step(qh, h, m)
    k1 = qg2_tendency(qh, m)
    k2 = qg2_tendency(qh .+ (h / 2) .* k1, m)
    k3 = qg2_tendency(qh .+ (h / 2) .* k2, m)
    k4 = qg2_tendency(qh .+ h .* k3, m)
    return (qh .+ (h / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)) .* m.filt
end

"pyqg와 같은 초기 PV (작은 무작위, 층 1만)."
function qg2_initial(m, rng)
    q1 = 1e-7 .* rand(rng, m.nx, m.nx) .+ 1e-6 .* (rand(rng, m.nx) * ones(1, m.nx))
    return cat(m.prf * q1, m.prf * zeros(m.nx, m.nx); dims=3)
end

"ψ̂ → 관측 벡터 (층 1, 층 2 순서; 각 모드 실부·허부; 진폭 = 물리 ψ의 Fourier 계수 = ψ̂ / nx²)."
function qg2_observe(ph, m)
    s = 1 / m.nx^2; o = Vector{Float64}(undef, m.nobs); c = 0
    for n in 1:2, (i, j) in m.oidx
        z = ph[i, j, n] * s
        o[c + 1] = real(z); o[c + 2] = imag(z); c += 2
    end
    return o
end
"관측 벡터 → ψ̂ (나머지 모드 0). k = 0 행의 l < 0은 켤레로 채운다 (실수장)."
function qg2_unobserve(o::AbstractVector, m)
    nk = m.nx ÷ 2 + 1; ph = zeros(ComplexF64, nk, m.nx, 2); c = 0
    for n in 1:2, (q, (i, j)) in enumerate(m.oidx)
        z = complex(o[c + 1], o[c + 2]) * m.nx^2; c += 2
        ph[i, j, n] = z
        m.obs[q][1] == 0 && (ph[1, mod(m.nx - (j - 1), m.nx) + 1, n] = conj(z))
    end
    return ph
end

"""
진단 (전체 해상도 ψ̂에서): 층 평균 운동 에너지 KE = Σ δ_n ½⟨|∇ψ_n|²⟩, 위치 에너지 APE = ½ δ1 F1 ⟨(ψ1 − ψ2)²⟩,
띠 평균 u_n(y) = −∂y ψ̄_n (k = 0 행). rfft 대칭 가중 (k > 0 열은 두 번).
"""
function qg2_energy(ph, m)
    s = 1 / m.nx^4; w = ones(size(m.K2)); w[2:end, :] .= 2.0; m.nx % 2 == 0 && (w[end, :] .= 1.0)
    ke = [0.5 * s * sum(w .* m.K2 .* abs2.(view(ph, :, :, n))) for n in 1:2]
    ape = 0.5 * m.del1 * m.F1 * s * sum(w .* abs2.(view(ph, :, :, 1) .- view(ph, :, :, 2)))
    return (KE=m.del1 * ke[1] + m.del2 * ke[2], KE1=ke[1], KE2=ke[2], APE=ape)
end
function qg2_ubar(ph, m)
    u = zeros(m.nx, 2)
    for n in 1:2
        c = -(m.il) .* ph[1, :, n] ./ m.nx                       # k = 0 행의 −i l ψ̂ → x 평균의 ifft (y 방향)
        u[:, n] = real.(ifft(c))
    end
    return u
end

"관측 벡터 X (모델 좌표, 열 = 시각)의 signature 계열: 대규모 운동 에너지, 위치 에너지, jet 세기 max_y ū1."
function qg2_series(m, X::AbstractMatrix)
    T = size(X, 2); ke = zeros(T); ape = zeros(T); jet = zeros(T)
    for t in 1:T
        ph = qg2_unobserve(view(X, :, t), m)
        e = qg2_energy(ph, m); ke[t] = e.KE; ape[t] = e.APE
        jet[t] = maximum(view(qg2_ubar(ph, m), :, 1))
    end
    return (KE=ke, APE=ape, jet=jet)
end

"관측 모드 라벨 (층, kx, ly, re/im)."
qg2_labels(m) = vec([string("ψ", n, "(", i, ",", j, ")", r) for r in ("re", "im"), (i, j) in m.obs, n in 1:2])

const _QG2_CACHE = Dict{UInt,Any}()
const QG2_PARAM_KEYS = sort(collect(keys(QG2_PARAM_SETS["eddy"])))
qg2_from_cfg(p::AbstractDict) = get!(() -> qg2_model(p), _QG2_CACHE, hash([(k, p[k]) for k in QG2_PARAM_KEYS]))
