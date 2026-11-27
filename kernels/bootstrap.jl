# Reproduction-only loader: no historical sweeps, messaging, GUI or original workspace.
ENV["HEADLESS"]="true"
using SHA,TOML,Dates,DelimitedFiles,Serialization,SparseArrays
include(joinpath(@__DIR__,"../src/engine/resync_engine.jl"))
include(joinpath(@__DIR__,"../src/systems/system_specs.jl"))
include("csv_writer.jl")
include("physics_helpers.jl")
include("quick_highdim_systems.jl")
include(joinpath(@__DIR__,"../src/systems/low_order_qg.jl"))
include("highdim_physics.jl")
include("resync_v4_evaluator.jl")
include("resync_v4_calibration.jl")
include("resync_v4_bo.jl")
include("resync_v4_operational.jl")
using .ResyncV4Metrics,.ResyncV4Evaluator
BLAS.set_num_threads(1)
