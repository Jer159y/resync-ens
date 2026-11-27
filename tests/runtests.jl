for name in ["resync_v4_metrics.jl","resync_v4_evaluator.jl","resync_v4_calibration.jl","resync_v4_bo.jl"]
    run(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --startup-file=no $(joinpath(@__DIR__,name))`)
end
