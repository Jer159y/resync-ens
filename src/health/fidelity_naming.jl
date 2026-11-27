# Terminology migration only: legacy member-test pass -> fidelity.
# Numeric class codes, thresholds and raw model output are unchanged.
fidelity_name(s::AbstractString) = replace(replace(replace(replace(String(s),
    "unhealthy" => "non_fidelity"), "HEALTHY" => "FIDELITY"), "Healthy" => "Fidelity"), "healthy" => "fidelity")
fidelity_name(s::Symbol) = Symbol(fidelity_name(String(s)))
fidelity_name(x) = x

fidelity_names(x) = x
fidelity_names(x::Symbol) = fidelity_name(x)
fidelity_names(x::AbstractString) = occursin(r"^[A-Za-z][A-Za-z_0-9]*$", x) ? fidelity_name(x) : x
function fidelity_names(x::AbstractDict)
    out = Dict{Any,Any}()
    for (k, v) in x
        key, value = fidelity_name(k), fidelity_names(v)
        haskey(out, key) && !isequal(out[key], value) && error("Conflicting legacy/fidelity keys: $key")
        out[key] = value
    end
    return out
end
function fidelity_names(x::NamedTuple)
    keys_new = Tuple(fidelity_name(k) for k in keys(x))
    length(unique(keys_new)) == length(keys_new) || error("Conflicting legacy/fidelity NamedTuple keys")
    return NamedTuple{keys_new}(map(fidelity_names, values(x)))
end
fidelity_names(x::Tuple) = map(fidelity_names, x)
fidelity_names(x::AbstractArray{<:Number}) = x
fidelity_names(x::AbstractArray) = map(fidelity_names, x)

"Read legacy keys and state symbols in memory; never rewrites archived JLD2 files."
function load_fidelity(args...)
    mapped = map(a -> a isa AbstractString ? String(a) : a, args)
    # Variable selection must use the stored spelling, so normalize after a full load.
    if length(mapped) == 2 && mapped[2] isa AbstractString
        data = fidelity_names(JLD2.load(mapped[1]))
        return data[fidelity_name(mapped[2])]
    end
    return fidelity_names(JLD2.load(mapped...))
end

"Legacy CSV/summary identifiers only; preserve paths and prose in archived records."
read_fidelity_lines(path) = [join(fidelity_names.(split(line, ',')), ',') for line in readlines(path)]
