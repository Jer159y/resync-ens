# Reproduce the original within-architecture Duffing validation example settings.
using TOML
kind=first(ARGS);kind in ["lstm","esn"]||error("Usage: sequential.jl lstm|esn [--test-only] [--retrain]")
root=normpath(joinpath(@__DIR__,".."));job=TOML.parsefile(joinpath(root,"configs/sequential_$kind.toml"))
out=abspath(get(ENV,"RESYNC_ENS_RUNS","runs"));inputs=abspath(get(ENV,"RESYNC_ENS_INPUTS","artifacts/inputs"))
job["output"]=joinpath(out,"sequential",kind,"p10");job["baseline_root"]=joinpath(out,"sequential",kind,"baseline")
job["model_root"]="--retrain" in ARGS ? joinpath(out,"sequential",kind,"retrained_models") : joinpath(inputs,"sequential",kind,"models")
if !("--test-only" in ARGS) && !("--retrain" in ARGS)
    all(isfile(joinpath(job["model_root"],"member_$i.bin")) for i in 1:300)||error("Missing frozen sequential models; fetch the input artifact")
end
mkpath(joinpath(out,"sequential",kind));path=joinpath(out,"sequential",kind,"job.toml")
open(io->TOML.print(io,job;sorted=true),path,"w");ARGS[1]=path
include(joinpath(root,"kernels/sequential_validation.jl"))
check2()
"--test-only" in ARGS||main_expanded()
