
using Statistics
using LinearAlgebra
using FFTW
using CairoMakie

# ============================================================
# Segment-wise diagnostics for periodic true-trajectory re-warmup
# ============================================================

"""
Global normalization scale computed once from the full truth trajectory.

Using one fixed scale is important when comparing different segments:
segment-wise variance normalization can make identical absolute errors receive
different NMSE values merely because a segment has smaller/larger variance.
"""
function global_truth_variance(truth::AbstractMatrix)
    return vec(var(truth; dims=2, corrected=false)) .+ eps(Float64)
end

"""
Pointwise MSE averaged over state dimensions.
"""
function pointwise_mse(
    truth::AbstractMatrix,
    pred::AbstractMatrix,
)
    size(truth) == size(pred) ||
        throw(DimensionMismatch("truth and pred must have the same size"))

    return vec(mean(abs2.(truth .- pred); dims=1))
end

"""
Pointwise NMSE using a fixed full-trajectory variance scale.
"""
function pointwise_nmse(
    truth::AbstractMatrix,
    pred::AbstractMatrix;
    variance_scale::AbstractVector=global_truth_variance(truth),
)
    size(truth) == size(pred) ||
        throw(DimensionMismatch("truth and pred must have the same size"))

    length(variance_scale) == size(truth, 1) ||
        throw(DimensionMismatch("variance_scale has wrong state dimension"))

    scaled_sqerr = abs2.(truth .- pred) ./ reshape(variance_scale, :, 1)
    return vec(mean(scaled_sqerr; dims=1))
end

"""
Valid prediction time measured inside one segment.

Returns a value in 0:length(segment).  If the first predicted point already
exceeds the threshold, the result is 0.
"""
function segment_vpt(
    truth_seg::AbstractMatrix,
    pred_seg::AbstractMatrix;
    threshold::Real=0.4,
    variance_scale::AbstractVector=global_truth_variance(truth_seg),
)
    nmse = pointwise_nmse(
        truth_seg,
        pred_seg;
        variance_scale=variance_scale,
    )

    first_bad = findfirst(>(threshold), nmse)
    return isnothing(first_bad) ? length(nmse) : first_bad - 1
end

"""
Length-normalized log power-spectrum distance.

Unlike an unnormalized L2 norm over FFT bins, this quantity does not grow
automatically just because the segment contains more frequency bins.
"""
function normalized_log_ps_distance(
    truth_seg::AbstractMatrix,
    pred_seg::AbstractMatrix,
)
    size(truth_seg) == size(pred_seg) ||
        throw(DimensionMismatch("truth_seg and pred_seg must have the same size"))

    T = size(truth_seg, 2)
    T >= 4 || return NaN

    ps_truth = abs2.(rfft(truth_seg, 2)) ./ T
    ps_pred  = abs2.(rfft(pred_seg, 2)) ./ T

    # Average over state dimensions, then compare per frequency bin.
    p = vec(mean(ps_truth; dims=1))
    q = vec(mean(ps_pred; dims=1))

    return sqrt(mean(abs2, log1p.(p) .- log1p.(q)))
end

"""
Average 1D Wasserstein distance over state dimensions.
"""
function segment_wasserstein(
    truth_seg::AbstractMatrix,
    pred_seg::AbstractMatrix,
)
    size(truth_seg) == size(pred_seg) ||
        throw(DimensionMismatch("truth_seg and pred_seg must have the same size"))

    return mean(
        DynamicsMetrics.evaluate(
            DynamicsMetrics.WassersteinDistance(reduction=:state),
            truth_seg,
            pred_seg,
        ).values,
    )
end

"""
1D Wasserstein distance, truncating to the shorter vector's length (sorted
first, so this keeps each side's smallest `n` values rather than its first `n`
in original order). Delegates to `DynamicsMetrics.WassersteinDistance`, which
requires equal-length input — the truncation above guarantees that.
"""
function wasserstein_1d(x::AbstractVector, y::AbstractVector)
    n = min(length(x), length(y))
    n == 0 && return NaN
    xs = sort(vec(x))[1:n]
    ys = sort(vec(y))[1:n]
    return DynamicsMetrics.evaluate(
        DynamicsMetrics.WassersteinDistance(reduction=:global), xs, ys,
    ).value
end

"""
Compute one NamedTuple of diagnostics for each segment.

The same full-trajectory variance scale is reused in every segment so that
segment NMSE values are directly comparable.
"""
function compute_segment_diagnostics(
    truth::AbstractMatrix,
    pred::AbstractMatrix;
    segment_len::Int,
    vpt_threshold::Real=0.4,
    variance_scale::AbstractVector=global_truth_variance(truth),
)
    size(truth) == size(pred) ||
        throw(DimensionMismatch("truth and pred must have the same size"))
    segment_len > 0 || throw(ArgumentError("segment_len must be positive"))

    T = size(truth, 2)
    n_segments = cld(T, segment_len)
    diagnostics = NamedTuple[]

    for seg in 1:n_segments
        first_t = (seg - 1) * segment_len + 1
        last_t  = min(seg * segment_len, T)
        r = first_t:last_t

        truth_seg = @view truth[:, r]
        pred_seg  = @view pred[:, r]

        mse_t  = pointwise_mse(truth_seg, pred_seg)
        nmse_t = pointwise_nmse(
            truth_seg,
            pred_seg;
            variance_scale=variance_scale,
        )

        vpt = segment_vpt(
            truth_seg,
            pred_seg;
            threshold=vpt_threshold,
            variance_scale=variance_scale,
        )

        push!(diagnostics, (
            segment=seg,
            range=r,
            length=length(r),
            mse=mean(mse_t),
            rmse=sqrt(mean(mse_t)),
            nmse=mean(nmse_t),
            final_cumulative_nmse=mean(nmse_t),
            vpt=vpt,
            vpt_fraction=vpt / length(r),
            ps_dist=normalized_log_ps_distance(truth_seg, pred_seg),
            wasserstein=segment_wasserstein(truth_seg, pred_seg),
            mse_time=mse_t,
            nmse_time=nmse_t,
            cumulative_nmse=cumsum(nmse_t) ./ (1:length(nmse_t)),
        ))
    end

    return diagnostics
end

function _finite_max(vectors...)
    values = Float64[]
    for v in vectors
        append!(values, filter(isfinite, Float64.(v)))
    end
    return isempty(values) ? 1.0 : maximum(values)
end

"""
Detailed segment grid.

Rows = segments.
Columns =
1. phase portrait,
2. first coordinate x₁(t),
3. pointwise NMSE,
4. cumulative NMSE inside the segment,
5. x₁ density histogram (truth vs prediction) + W₁,
6. x₂ density histogram (truth vs prediction) + W₁.

For a very long trajectory, use `max_segments` or `segment_indices` so that
the figure remains readable.
"""
function plot_segment_detail_grid(
    truth::AbstractMatrix,
    pred::AbstractMatrix;
    segment_len::Int,
    method_name::AbstractString="Periodic re-warmup",
    vpt_threshold::Real=0.4,
    max_segments::Int=12,
    segment_indices=nothing,
    save_path::Union{Nothing,AbstractString}=nothing,
)
    size(truth, 1) >= 2 ||
        throw(ArgumentError("phase portrait requires at least two state dimensions"))

    variance_scale = global_truth_variance(truth)
    diagnostics = compute_segment_diagnostics(
        truth,
        pred;
        segment_len=segment_len,
        vpt_threshold=vpt_threshold,
        variance_scale=variance_scale,
    )

    all_indices = collect(eachindex(diagnostics))
    selected = if segment_indices === nothing
        all_indices[1:min(max_segments, length(all_indices))]
    else
        collect(segment_indices)
    end

    all(i -> 1 <= i <= length(diagnostics), selected) ||
        throw(BoundsError(diagnostics, selected))

    phase_x = vcat(vec(truth[1, :]), vec(pred[1, :]))
    phase_y = vcat(vec(truth[2, :]), vec(pred[2, :]))
    x1_all  = vcat(vec(truth[1, :]), vec(pred[1, :]))

    max_nmse = _finite_max(
        [d.nmse_time for d in diagnostics]...,
        [d.cumulative_nmse for d in diagnostics]...,
    )
    max_nmse = max(max_nmse, Float64(vpt_threshold))

    fig = Figure(size=(2300, 300 * length(selected) + 80))

    Label(
        fig[0, 1:6],
        "$(method_name): segment-wise diagnostics (segment length = $(segment_len))";
        fontsize=24,
        font=:bold,
    )

    for (plot_row, seg_idx) in enumerate(selected)
        d = diagnostics[seg_idx]
        r = d.range
        local_t = 1:d.length

        # 1. Phase portrait
        ax_phase = Axis(
            fig[plot_row, 1];
            title="Segment $(seg_idx): t=$(first(r)):$(last(r))",
            xlabel="x₁",
            ylabel="x₂",
        )
        lines!(ax_phase, truth[1, r], truth[2, r]; label="Truth", linewidth=2)
        lines!(ax_phase, pred[1, r], pred[2, r]; label="Prediction", linewidth=1.5)
        xlims!(ax_phase, extrema(phase_x)...)
        ylims!(ax_phase, extrema(phase_y)...)
        axislegend(ax_phase; position=:rb)

        # 2. First coordinate trajectory
        ax_traj = Axis(
            fig[plot_row, 2];
            title="x₁ trajectory",
            xlabel="Local step",
            ylabel="x₁",
        )
        lines!(ax_traj, local_t, truth[1, r]; label="Truth", linewidth=2)
        lines!(ax_traj, local_t, pred[1, r]; label="Prediction", linewidth=1.5)
        ylims!(ax_traj, extrema(x1_all)...)

        # 3. Pointwise NMSE
        ax_nmse = Axis(
            fig[plot_row, 3];
            title="Pointwise NMSE · VPT=$(d.vpt)/$(d.length)",
            xlabel="Local step",
            ylabel="NMSE",
        )
        lines!(ax_nmse, local_t, d.nmse_time)
        hlines!(
            ax_nmse,
            [vpt_threshold];
            linestyle=:dash,
            linewidth=1.2,
            label="VPT threshold",
        )
        ylims!(ax_nmse, 0, max_nmse * 1.05)
        axislegend(ax_nmse; position=:rt)

        # 4. Cumulative NMSE
        ax_cum = Axis(
            fig[plot_row, 4];
            title="Cumulative NMSE · final=$(round(d.nmse; sigdigits=4))",
            xlabel="Local step",
            ylabel="Mean NMSE up to step",
        )
        lines!(ax_cum, local_t, d.cumulative_nmse)
        ylims!(ax_cum, 0, max_nmse * 1.05)

        truth_x1 = vec(truth[1, r])
        pred_x1 = vec(pred[1, r])
        truth_x2 = vec(truth[2, r])
        pred_x2 = vec(pred[2, r])

        w1_x1 = wasserstein_1d(truth_x1, pred_x1)
        w1_x2 = wasserstein_1d(truth_x2, pred_x2)

        # 5. x1 density histogram
        ax_hist1 = Axis(
            fig[plot_row, 5];
            title="x₁ density · W₁=$(round(w1_x1, digits=4))",
            xlabel="x₁",
            ylabel="Density",
        )
        hist!(ax_hist1, truth_x1; bins=50, normalization=:pdf, alpha=0.45, label="Truth")
        hist!(ax_hist1, pred_x1; bins=50, normalization=:pdf, alpha=0.45, label="Prediction")
        plot_row == 1 && axislegend(ax_hist1; position=:rt)

        # 6. x2 density histogram
        ax_hist2 = Axis(
            fig[plot_row, 6];
            title="x₂ density · W₁=$(round(w1_x2, digits=4))",
            xlabel="x₂",
            ylabel="Density",
        )
        hist!(ax_hist2, truth_x2; bins=50, normalization=:pdf, alpha=0.45, label="Truth")
        hist!(ax_hist2, pred_x2; bins=50, normalization=:pdf, alpha=0.45, label="Prediction")
        plot_row == 1 && axislegend(ax_hist2; position=:rt)
    end

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 18)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig, diagnostics
end

"""
Compare segment-level metrics between two predictions.

Typical use:
- prediction A = continuous autonomous rollout
- prediction B = periodic true-trajectory re-warmup

Rows/columns are chosen as a 2×3 metric dashboard rather than one lollipop
per row, because the five metrics have incompatible numerical scales.
"""
function plot_segment_metric_evolution(
    truth::AbstractMatrix,
    predictions::AbstractVector{<:AbstractMatrix},
    labels::AbstractVector{<:AbstractString};
    segment_len::Int,
    vpt_threshold::Real=0.4,
    save_path::Union{Nothing,AbstractString}=nothing,
)
    length(predictions) == length(labels) ||
        throw(DimensionMismatch("predictions and labels must have equal lengths"))

    variance_scale = global_truth_variance(truth)
    all_diagnostics = [
        compute_segment_diagnostics(
            truth,
            pred;
            segment_len=segment_len,
            vpt_threshold=vpt_threshold,
            variance_scale=variance_scale,
        )
        for pred in predictions
    ]

    metric_specs = [
        (:vpt_fraction, "VPT fraction", "VPT / segment length"),
        (:nmse, "Mean NMSE", "NMSE"),
        (:rmse, "RMSE", "RMSE"),
        (:ps_dist, "Normalized log-PS distance", "PS distance"),
        (:wasserstein, "Wasserstein distance", "W₁"),
    ]

    fig = Figure(size=(1400, 850))
    Label(
        fig[0, 1:3],
        "Segment-wise metric evolution (segment length = $(segment_len))";
        fontsize=24,
        font=:bold,
    )

    for (panel, (key, title, ylabel)) in enumerate(metric_specs)
        row = panel <= 3 ? 1 : 2
        col = panel <= 3 ? panel : panel - 3

        ax = Axis(
            fig[row, col];
            title=title,
            xlabel="Segment index",
            ylabel=ylabel,
        )

        for (diagnostics, label) in zip(all_diagnostics, labels)
            x = [d.segment for d in diagnostics]
            y = [getfield(d, key) for d in diagnostics]
            lines!(ax, x, y; label=label, linewidth=2)
            scatter!(ax, x, y; markersize=7)
        end

        key == :vpt_fraction && ylims!(ax, 0, 1.05)
        panel == 1 && axislegend(ax; position=:rt)
    end

    # Last panel: relative NMSE gain of the final method against the first.
    ax_gain = Axis(
        fig[2, 3];
        title="NMSE ratio",
        xlabel="Segment index",
        ylabel="$(labels[end]) / $(labels[1])",
    )

    baseline = [d.nmse for d in first(all_diagnostics)]
    candidate = [d.nmse for d in last(all_diagnostics)]
    n = min(length(baseline), length(candidate))
    ratio = candidate[1:n] ./ max.(baseline[1:n], eps(Float64))

    lines!(ax_gain, 1:n, ratio; linewidth=2)
    scatter!(ax_gain, 1:n, ratio; markersize=7)
    hlines!(ax_gain, [1.0]; linestyle=:dash, linewidth=1.2)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig, all_diagnostics
end

"""
Run periodic re-warmup for several segment lengths and collect both global
and per-segment diagnostics.

`predictor` must be a function accepting one integer segment length and
returning the corresponding prediction matrix.
"""
function evaluate_segment_lengths(
    truth::AbstractMatrix,
    segment_lengths::AbstractVector{Int},
    predictor::Function;
    vpt_threshold::Real=0.4,
)
    variance_scale = global_truth_variance(truth)
    outputs = Dict{Int, Matrix{Float64}}()
    diagnostics = Dict{Int, Vector{NamedTuple}}()
    summaries = NamedTuple[]

    for L in segment_lengths
        pred = Matrix{Float64}(predictor(L))
        size(pred) == size(truth) ||
            throw(DimensionMismatch("prediction for segment length $L has wrong size"))

        d = compute_segment_diagnostics(
            truth,
            pred;
            segment_len=L,
            vpt_threshold=vpt_threshold,
            variance_scale=variance_scale,
        )

        outputs[L] = pred
        diagnostics[L] = d

        all_nmse = pointwise_nmse(
            truth,
            pred;
            variance_scale=variance_scale,
        )
        all_mse = pointwise_mse(truth, pred)

        push!(summaries, (
            segment_len=L,
            mean_segment_vpt_fraction=mean(x.vpt_fraction for x in d),
            median_segment_vpt_fraction=median(x.vpt_fraction for x in d),
            global_nmse=mean(all_nmse),
            global_rmse=sqrt(mean(all_mse)),
            mean_segment_ps_dist=mean(x.ps_dist for x in d if isfinite(x.ps_dist)),
            mean_segment_wasserstein=mean(x.wasserstein for x in d),
        ))
    end

    sort!(summaries; by=x -> x.segment_len)
    return outputs, diagnostics, summaries
end

"""
Plot the effect of re-warmup segment length.

Each panel uses its own y-axis because VPT fraction, NMSE, RMSE, spectral
distance, and Wasserstein distance are not commensurate.
"""
function plot_segment_length_comparison(
    summaries::AbstractVector{<:NamedTuple};
    save_path::Union{Nothing,AbstractString}=nothing,
)
    ordered = sort(collect(summaries); by=x -> x.segment_len)
    x = [s.segment_len for s in ordered]

    specs = [
        (:mean_segment_vpt_fraction, "Mean segment VPT fraction", "VPT / segment length"),
        (:global_nmse, "Global NMSE", "NMSE"),
        (:global_rmse, "Global RMSE", "RMSE"),
        (:mean_segment_ps_dist, "Mean normalized log-PS distance", "PS distance"),
        (:mean_segment_wasserstein, "Mean Wasserstein distance", "W₁"),
    ]

    fig = Figure(size=(1400, 850))
    Label(
        fig[0, 1:3],
        "Periodic re-warmup: comparison across segment lengths";
        fontsize=24,
        font=:bold,
    )

    for (panel, (key, title, ylabel)) in enumerate(specs)
        row = panel <= 3 ? 1 : 2
        col = panel <= 3 ? panel : panel - 3

        ax = Axis(
            fig[row, col];
            title=title,
            xlabel="Segment length (steps)",
            ylabel=ylabel,
            xscale=log10,
        )

        y = [getfield(s, key) for s in ordered]
        lines!(ax, x, y; linewidth=2)
        scatter!(ax, x, y; markersize=10)

        for (xi, yi) in zip(x, y)
            text!(ax, xi, yi; text=string(round(yi; sigdigits=4)), align=(:center, :bottom))
        end

        key == :mean_segment_vpt_fraction && ylims!(ax, 0, 1.05)
    end

    # Sixth panel: textual recommendation table.
    ax_text = Axis(fig[2, 3])
    hidedecorations!(ax_text)
    hidespines!(ax_text)

    best_nmse = ordered[argmin([s.global_nmse for s in ordered])]
    best_vpt  = ordered[argmax([s.mean_segment_vpt_fraction for s in ordered])]
    best_ps   = ordered[argmin([s.mean_segment_ps_dist for s in ordered])]

    text!(
        ax_text,
        0.02,
        0.95;
        space=:relative,
        align=(:left, :top),
        text=(
            "Best by metric\n\n" *
            "Global NMSE: L=$(best_nmse.segment_len)\n" *
            "Mean VPT fraction: L=$(best_vpt.segment_len)\n" *
            "Spectral distance: L=$(best_ps.segment_len)\n\n" *
            "Choose L from the metric that\n" *
            "matches the scientific objective."
        ),
        fontsize=18,
    )

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end
