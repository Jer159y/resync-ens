isdefined(@__MODULE__, :fidelity_names) || include(joinpath(@__DIR__, "health", "fidelity_naming.jl"))
# prelude.jl — the start of every script: packages, Telegram alarm, ESN utilities (module TestUtils), progress-safe logging.
const SRC_DIR = @__DIR__
include(joinpath(SRC_DIR, "core", "packages.jl"))

include(joinpath(SRC_DIR, "core", "esn_utils.jl"))

# ProgressMeter's `next!`/`@showprogress` redraw a bar in place via a bare "\r" +
# ANSI clear-to-end-of-line, with no trailing newline in between frames. Any
# @info/@warn/@error fired while a bar is live (from this repo's own code, or
# from ReservoirComputing/DifferentialEquations internals) lands mid-line and
# gets glued onto the bar's last-drawn frame instead of starting on a fresh
# line. Wrapping the global logger to clear the current stdout line first
# fixes this regardless of which package/call site logs — this mirrors what
# `printover` itself already does, so it composes safely even when nothing is
# mid-line (clearing an already-clear line is a no-op).
using Logging: Logging, current_logger, global_logger

struct ProgressSafeLogger{L<:Logging.AbstractLogger} <: Logging.AbstractLogger
    inner::L
end

Logging.min_enabled_level(l::ProgressSafeLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::ProgressSafeLogger, level, _module, group, id) =
    Logging.shouldlog(l.inner, level, _module, group, id)
Logging.catch_exceptions(l::ProgressSafeLogger) = Logging.catch_exceptions(l.inner)

function Logging.handle_message(l::ProgressSafeLogger, level, message, _module, group, id, file, line; kwargs...)
    print(stdout, "\r\u1b[K")
    flush(stdout)
    return Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
end

current_logger() isa ProgressSafeLogger || global_logger(ProgressSafeLogger(current_logger()))
