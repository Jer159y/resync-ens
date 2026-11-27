# Optional from-equation data/calibration check; do not replace frozen input artifacts.
include("common.jl")
a=case_inputs(first(ARGS));p=a.spec;m=a.manifest;seed=p["physical_seeds"]["calibration_physical_seed"]
N=m["calibration_indices"]["check_targets"]["last"]
X=haskey(p,"physics") ? highdim_fresh(a.cfg,N,seed,p["physics"]) : scheme_fresh(a.cfg,N,seed,p["physical_design"])
Z=(X.-a.mu)./a.sd
slicer(key)=begin r=m["calibration_indices"][key];view(Z,:,r["first"]:r["last"]) end
maximum(abs.(slicer("w1_target_bank").-a.target))<1e-10||error("Regenerated calibration target differs from archive")
c=p["calibration"];num=p["numerics"]
r=ResyncV4Calibration.calibrate(slicer("w1_target_bank"),slicer("population_window_pool"),slicer("check_targets"),p["base_tau"],300;
    fit_draws=c["base_window_draws"],check_draws=c["base_window_draws"],distribution_draws=c["distribution_window_draws"],seed=c["seed"],
    dimension_samples=num["effective_dimension_sample_times"],constant_tol=num["constant_amplitude_tolerance"],numerical_floor=num["positive_loss_numerical_floor"],progress=true)
for key in ["spread_reference","floor_w","floor_s"]
    isapprox(getproperty(r,Symbol(key)),a.reference[key];atol=1e-10,rtol=1e-10)||error("Reference mismatch: $key")
end
out=joinpath(RUN_ROOT,a.id,"regenerated_calibration");mkpath(out);JLD2.jldsave(joinpath(out,"reference.jld2");reference=r)
println("CALIBRATION_REPRODUCED $(a.id)")
