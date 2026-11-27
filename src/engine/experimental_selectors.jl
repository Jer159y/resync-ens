# Experimental blind selectors. No test truth or health labels enter these scores.
const LOOKAHEAD_SELECTORS = (:trial_repair, :peer_check)

function selector_shortlist(pool, ctx)
    c = crit_feature_distance_from_centroid(pool, ctx)
    ids = findall(pool.fin)
    sort!(ids; by=i -> c[i], rev=true)
    K = clamp(round(Int, ctx.selector_runtime.p * length(ids)), 1, max(1, length(ids) - 1))
    return ids[1:min(2K, length(ids))], reverse(ids), c
end

function selector_history(pool, ctx, i)
    rt = ctx.selector_runtime
    pool.s == 1 && return pool.cands[i]
    return hcat(assemble_history(rt.hist, rt.src, i, pool.s - 1, ctx.τ), pool.cands[i])
end

function selector_train_distance(ctx, Y)
    all(isfinite, Y) || return Inf
    z = pca_project(ctx.pca, fe1(Y); k=min(8, ctx.pca.k_crit))
    return all(isfinite, z) ? norm(z) : Inf
end

# Audition a full-history repair using a provisional low-centroid-score donor.
# Actual round-robin donor assignment can differ after the final shortlist is chosen.
function crit_trial_repair(pool::SegmentPool, ctx)
    scores = [f ? -Inf : Inf for f in pool.fin]
    candidates, donors, _ = selector_shortlist(pool, ctx)
    isempty(donors) && return scores
    Threads.@threads for q in eachindex(candidates)
        i = candidates[q]
        ds = [j for j in donors[1:max(1, length(donors) ÷ 2)] if j != i]
        isempty(ds) && continue
        d = ds[mod1(i, length(ds))]
        ps, _, esn = ctx.ensemble[i]
        own, _ = ReservoirComputing.predict(esn, ctx.τ, ps, deepcopy(ctx.selector_runtime.st_auto[i]); initialdata=pool.cands[i][:, end])
        Y = selector_history(pool, ctx, d)
        st = warmup_full_history(ps, ctx.init_states[i], esn, Y; x0=ctx.x0)
        repaired, _ = ReservoirComputing.predict(esn, ctx.τ, ps, st; initialdata=Y[:, end])
        a = selector_train_distance(ctx, own); b = selector_train_distance(ctx, repaired)
        scores[i] = isfinite(b) ? a - b : -Inf
    end
    return scores
end

# Three low-centroid-score peers predict the final short tail of each candidate.
# Their states are synchronised on up to 2 tau of that candidate's available history.
function crit_peer_check(pool::SegmentPool, ctx)
    scores = [f ? -Inf : Inf for f in pool.fin]
    candidates, peers, _ = selector_shortlist(pool, ctx)
    h = min(10, max(2, ctx.τ ÷ 20))
    Threads.@threads for q in eachindex(candidates)
        i = candidates[q]
        judges = [j for j in peers if j != i][1:min(3, length(peers) - 1)]
        isempty(judges) && continue
        Y = selector_history(pool, ctx, i)
        stop = size(Y, 2) - h
        prefix = Y[:, max(1, stop - 2ctx.τ + 1):stop]
        target = Y[:, (stop + 1):end]
        errs = Float64[]
        for j in judges
            ps, _, esn = ctx.ensemble[j]
            st = warmup_full_history(ps, ctx.init_states[j], esn, prefix; x0=ctx.x0)
            pred, _ = ReservoirComputing.predict(esn, h, ps, st; initialdata=prefix[:, end])
            push!(errs, all(isfinite, pred) ? sqrt(mean(abs2, pred .- target)) : Inf)
        end
        scores[i] = median(errs)
    end
    return scores
end

# Leave-one-out influence on the ensemble FE1 PCA mean and diagonal second moment.
# Training PCA has zero mean and eigenvalues ratio * number of standardised features.
function crit_distribution_influence(pool::SegmentPool, ctx)
    scores = fill(Inf, length(pool.cands))
    ids = findall(pool.fin); length(ids) > 1 || return scores
    k = min(8, ctx.pca.k_crit)
    Z = reduce(hcat, [pca_project(ctx.pca, fe1(pool.cands[i]); k=k) for i in ids])
    target = ctx.pca.ratio[1:k] .* length(ctx.pca.mu)
    loss(mu, m2) = sum(abs2, mu) + 0.25sum(abs2, (m2 .- target) ./ (target .+ 0.1))
    n = length(ids); a = vec(sum(Z; dims=2)); b = vec(sum(abs2, Z; dims=2))
    before = loss(a ./ n, b ./ n)
    for (q, i) in enumerate(ids)
        scores[i] = before - loss((a .- Z[:, q]) ./ (n - 1), (b .- Z[:, q].^2) ./ (n - 1))
    end
    return scores
end
