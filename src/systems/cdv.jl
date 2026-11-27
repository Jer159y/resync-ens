# ============================================================
# cdv.jl — Charney–DeVore 6모드 모델 (Charney & DeVore 1979; Crommelin, Opsteegh & Verhulst 2004, JAS 61, 1406)
#
# β-평면 채널의 순압 흐름 + 지형 (고도 진폭 γ), 띠 평균 흐름 x1, x4 (cos y, cos 2y)와 파수 1, 2 파동 (x2, x3), (x5, x6):
#   ẋ1 = γ̃1 x3 − C(x1 − x1*)
#   ẋ2 = −(α1 x1 − β1) x3 − C x2 − δ1 x4 x6
#   ẋ3 = (α1 x1 − β1) x2 − γ1 x1 − C x3 + δ1 x4 x5
#   ẋ4 = γ̃2 x6 − C(x4 − x4*) + ε(x2 x6 − x3 x5)
#   ẋ5 = −(α2 x1 − β2) x6 − C x5 − δ2 x4 x3
#   ẋ6 = (α2 x1 − β2) x5 − γ2 x4 − C x6 + δ2 x4 x2
#   α_m = 8√2 m²(b² + m² − 1) / (π(4m² − 1)(b² + m²)),   β_m = β b² / (b² + m²),
#   δ_m = 64√2 (b² − m² + 1) / (15π(b² + m²)),         γ̃_m = γ 4√2 m b / (π(4m² − 1)),
#   γ_m = γ 4√2 m³ b / (π(4m² − 1)(b² + m²)),          ε = 16√2 / (5π)
# 계수는 NPG 30, 49 (2023)의 식 (1)과 SSMTool examples/CharneyDeVore1stOrder/build_model.m에서 대조.
# Crommelin 2004 레짐 설정: x1* = 0.95, x4* = −0.76095, C = 0.1 (감쇠 ~10일 → 시간 단위 ≈ 1일), β = 1.25, γ = 0.2, b = 0.5.
# 띠 흐름(zonal) 레짐과 막힘(blocked) 레짐을 오가는 전이가 특징 (x1–x4 평면).
# ============================================================

"CDV 계수 (cfg [system]의 x1s, x4s, C, beta, gamma, b)."
function cdv_coeffs(p::AbstractDict)
    b, β, γ = Float64(p["b"]), Float64(p["beta"]), Float64(p["gamma"])
    m = (1.0, 2.0)
    f(g) = map(g, m)
    return (C=Float64(p["C"]), x1s=Float64(p["x1s"]), x4s=Float64(p["x4s"]),
        α=f(m -> 8√2 * m^2 * (b^2 + m^2 - 1) / (π * (4m^2 - 1) * (b^2 + m^2))),
        β=f(m -> β * b^2 / (b^2 + m^2)),
        δ=f(m -> 64√2 * (b^2 - m^2 + 1) / (15π * (b^2 + m^2))),
        γt=f(m -> γ * 4√2 * m * b / (π * (4m^2 - 1))),
        γm=f(m -> γ * 4√2 * m^3 * b / (π * (4m^2 - 1) * (b^2 + m^2))),
        ε=16√2 / (5π))
end

function cdv_rhs!(du, x, c)
    x1, x2, x3, x4, x5, x6 = x
    du[1] = c.γt[1] * x3 - c.C * (x1 - c.x1s)
    du[2] = -(c.α[1] * x1 - c.β[1]) * x3 - c.C * x2 - c.δ[1] * x4 * x6
    du[3] = (c.α[1] * x1 - c.β[1]) * x2 - c.γm[1] * x1 - c.C * x3 + c.δ[1] * x4 * x5
    du[4] = c.γt[2] * x6 - c.C * (x4 - c.x4s) + c.ε * (x2 * x6 - x3 * x5)
    du[5] = -(c.α[2] * x1 - c.β[2]) * x6 - c.C * x5 - c.δ[2] * x4 * x3
    du[6] = (c.α[2] * x1 - c.β[2]) * x5 - c.γm[2] * x4 - c.C * x6 + c.δ[2] * x4 * x2
    return du
end

function cdv_rk4!(x, h, c, k1, k2, k3, k4, tmp)
    cdv_rhs!(k1, x, c); tmp .= x .+ (h / 2) .* k1
    cdv_rhs!(k2, tmp, c); tmp .= x .+ (h / 2) .* k2
    cdv_rhs!(k3, tmp, c); tmp .= x .+ h .* k3
    cdv_rhs!(k4, tmp, c); x .+= (h / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end
