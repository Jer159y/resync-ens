# ============================================================
# feature_extractors.jl
#
# Trajectory → feature-vector extractors, kept separate from any one
# Statement so they can be reused. Introduced by
# StatementC_feature_analysis.jl to replace the compression-reservoir
# feature vector (`compute_feature_vector` in
# ensemble/selection/embedding_features.jl) with cheaper, interpretable
# "raw" descriptors of a rollout segment.
#
# All extractors take a `traj::AbstractMatrix` shaped `(space_dim × T)`
# (state dimension along rows, time along columns) and return a
# `Vector{Float64}` feature vector. They do NOT reference any reservoir
# internals, training data, or ground truth — purely trajectory-shape
# statistics — so they are safe to use inside a blind selector.
#
#   FE-1  extract_stat_features      : per-dimension time-direction moments
#                                      (mean/std/skew/excess-kurt) + zero-crossing
#                                      rate + windowed local variance
#   FE-2  extract_fourier_features   : space-averaged power spectrum, top-n peak
#                                      magnitudes and their frequency-bin indices
#   FE-3  compute_leverage_scores    : SVD leverage score of every row of a
#                                      (n_samples × feature_dim) feature matrix —
#                                      "how extreme is this candidate in feature
#                                      space" (operates on a population, not one traj)
#   FE-4  extract_delay_features     : Takens delay embedding of the trajectory,
#                                      leading singular values as features
#
# Also exports the matching `*_feature_names(...)` helpers so plots can
# label columns.
# ============================================================

module FeatureExtractors

using Statistics
using LinearAlgebra
using FFTW

export extract_stat_features, extract_fourier_features,
       compute_leverage_scores, extract_delay_features,
       stat_feature_names, fourier_feature_names, delay_feature_names,
       zero_crossing_rate, local_variance

# ------------------------------------------------------------
# small moment helpers (population convention, std-0 tolerant)
# ------------------------------------------------------------

function _skewness(x::AbstractVector)
    σ = std(x; corrected=false)
    σ == 0 && return 0.0
    μ = mean(x)
    return mean(((x .- μ) ./ σ) .^ 3)
end

function _kurtosis(x::AbstractVector)   # excess kurtosis (normal → 0)
    σ = std(x; corrected=false)
    σ == 0 && return 0.0
    μ = mean(x)
    return mean(((x .- μ) ./ σ) .^ 4) - 3.0
end

# ------------------------------------------------------------
# FE-1: raw trajectory statistics
# ------------------------------------------------------------

"""
    zero_crossing_rate(traj::AbstractMatrix) -> Float64

Fraction of consecutive time steps at which the sign changes, averaged
over the `space_dim` rows. A crossing of exactly zero is counted when the
sign of `traj[d, t]` differs from `traj[d, t-1]` (treating `0` as its own
sign boundary via `< 0`).
"""
function zero_crossing_rate(traj::AbstractMatrix)
    D, T = size(traj)
    T < 2 && return 0.0
    total = 0.0
    @inbounds for d in 1:D
        c = 0
        prev_neg = traj[d, 1] < 0
        for t in 2:T
            cur_neg = traj[d, t] < 0
            cur_neg != prev_neg && (c += 1)
            prev_neg = cur_neg
        end
        total += c / (T - 1)
    end
    return total / D
end

"""
    local_variance(traj::AbstractMatrix; w::Int=10) -> Float64

Mean of the variance computed over every length-`w` sliding window (stride
1) of every row. A short-timescale "roughness" measure; contrast with the
global per-row variance in `extract_stat_features`.
"""
function local_variance(traj::AbstractMatrix; w::Int=10)
    D, T = size(traj)
    w = min(w, T)
    w < 2 && return 0.0
    acc = 0.0
    cnt = 0
    @inbounds for d in 1:D
        row = view(traj, d, :)
        for s in 1:(T - w + 1)
            acc += var(view(row, s:(s + w - 1)))
            cnt += 1
        end
    end
    return cnt == 0 ? 0.0 : acc / cnt
end

"""
    extract_stat_features(traj::AbstractMatrix; local_var_window=10, with_zero_crossing=true) -> Vector{Float64}

`traj` is `(space_dim × T)`. Returns, concatenated:
`[mean(1:D); std(1:D); skew(1:D); excess_kurt(1:D); (zero_crossing_rate;) local_variance]`
(length `4·space_dim + 2`, or `4·space_dim + 1` when `with_zero_crossing=false`), all along
the time direction. `with_zero_crossing=false` drops the zero-crossing-rate entry — used by
reservoir-level analyses that only want unbounded-scale shape statistics.
"""
function extract_stat_features(traj::AbstractMatrix; local_var_window::Int=10,
                               with_zero_crossing::Bool=true)
    D = size(traj, 1)
    μ = vec(mean(traj; dims=2))
    σ = vec(std(traj; dims=2))
    sk = [_skewness(view(traj, d, :)) for d in 1:D]
    kt = [_kurtosis(view(traj, d, :)) for d in 1:D]
    lv = local_variance(traj; w=local_var_window)
    tail = with_zero_crossing ? [zero_crossing_rate(traj), lv] : [lv]
    return Float64.(vcat(μ, σ, sk, kt, tail))
end

function stat_feature_names(D::Int; with_zero_crossing::Bool=true)
    return vcat(
        ["mean_x$d" for d in 1:D],
        ["std_x$d" for d in 1:D],
        ["skew_x$d" for d in 1:D],
        ["kurt_x$d" for d in 1:D],
        with_zero_crossing ? ["zero_cross_rate", "local_var"] : ["local_var"],
    )
end

# ------------------------------------------------------------
# FE-2: Fourier-based
# ------------------------------------------------------------

"""
    extract_fourier_features(traj::AbstractMatrix; n_peaks=5) -> Vector{Float64}

Space-averaged power spectrum (`mean` over rows of `abs2.(rfft(traj, 2))`),
then the magnitudes of the `n_peaks` largest spectral bins followed by
those bins' (1-based) frequency indices — length `2·n_peaks`. Peaks are
returned in descending-magnitude order.
"""
function extract_fourier_features(traj::AbstractMatrix; n_peaks::Int=5)
    ps = vec(mean(abs2.(rfft(traj, 2)); dims=1))
    k = min(n_peaks, length(ps))
    peak_idx = partialsortperm(ps, 1:k; rev=true)
    mags = Float64.(ps[peak_idx])
    freqs = Float64.(collect(peak_idx))
    if k < n_peaks                       # pad if the segment was extremely short
        mags = vcat(mags, zeros(n_peaks - k))
        freqs = vcat(freqs, zeros(n_peaks - k))
    end
    return vcat(mags, freqs)
end

fourier_feature_names(n_peaks::Int=5) = vcat(
    ["fft_peak$(i)_mag" for i in 1:n_peaks],
    ["fft_peak$(i)_freq" for i in 1:n_peaks],
)

# ------------------------------------------------------------
# FE-3: leverage score over a candidate population
# ------------------------------------------------------------

"""
    compute_leverage_scores(feature_matrix; rtol=1e-3, standardize=true) -> Vector{Float64}

`feature_matrix` is `(n_samples × feature_dim)`. Columns are z-scored
(unless `standardize=false`), a thin SVD is taken, the effective rank `k`
is `count(s > s₁·rtol)`, and each sample's leverage score is
`sum(U[i, 1:k] .^ 2)`. High leverage ⇒ that sample sits at an extreme /
isolated position in feature space.

Non-finite entries are replaced with `0` before the SVD so a diverged
candidate row cannot crash the decomposition (such rows simply score low).
"""
function compute_leverage_scores(feature_matrix::AbstractMatrix; rtol::Real=1e-3, standardize::Bool=true)
    F = Matrix{Float64}(feature_matrix)
    @inbounds for i in eachindex(F)
        isfinite(F[i]) || (F[i] = 0.0)
    end
    n = size(F, 1)
    n == 0 && return Float64[]
    if standardize
        μ = mean(F; dims=1)
        σ = std(F; dims=1)
        @inbounds for j in eachindex(σ)
            σ[j] == 0 && (σ[j] = 1.0)
        end
        F = (F .- μ) ./ σ
    end
    local U, S
    try
        F_svd = svd(F; full=false)
        U, S = F_svd.U, F_svd.S
    catch
        return fill(NaN, n)
    end
    isempty(S) && return zeros(n)
    k = max(1, count(>(S[1] * rtol), S))
    return vec(sum(abs2, @view(U[:, 1:k]); dims=2))
end

# ------------------------------------------------------------
# FE-4: delay embedding + SVD
# ------------------------------------------------------------

"""
    extract_delay_features(traj; delay=5, embed_dim=3, n_components=10) -> Vector{Float64}

Takens delay embedding: stack `embed_dim` copies of `traj`, each shifted
by `delay` steps, as extra rows (giving a `(space_dim·embed_dim × L)`
trajectory matrix, `L = T - delay·(embed_dim-1)`), then return the leading
`n_components` singular values of that matrix. If the segment is too short
for the embedding, returns a zero vector of length `n_components`.
"""
function extract_delay_features(traj::AbstractMatrix; delay::Int=5, embed_dim::Int=3, n_components::Int=10)
    T = size(traj, 2)
    span = delay * (embed_dim - 1)
    (T <= span + 1) && return zeros(Float64, n_components)
    L = T - span
    blocks = [traj[:, (1 + k):(L + k)] for k in 0:delay:span]
    embedded = reduce(vcat, blocks)
    local S
    try
        S = svd(embedded; full=false).S
    catch
        return zeros(Float64, n_components)
    end
    out = zeros(Float64, n_components)
    m = min(n_components, length(S))
    out[1:m] .= S[1:m]
    return out
end

delay_feature_names(n_components::Int=10) = ["delay_sv$i" for i in 1:n_components]

end # module FeatureExtractors
