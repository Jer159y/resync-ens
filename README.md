# resync-ens

Reproducibility materials for **State resynchronization in reservoir ensembles without additional observations** (Soobin Choi and Pilwon Kim), working GMD manuscript, export cutoff 28 November 2026.

This private repository contains the code and numerical evidence used by the manuscript, not the parent EESN development history. The main examples are the Duffing oscillator, Charney–DeVore model, Kuramoto–Sivashinsky equation (200-node reservoirs), a QG model with 20 variables, and a cyclic Thomas system with 16 variables.

## Fast reproduction: no forecast or model training

Python 3.11+ and a TeX installation with `pdflatex` and `bibtex`:

```sh
python3 scripts/verify.py
python3 scripts/reproduce_tables.py
python3 scripts/build.py
```

The paper is written to `build/paper/manuscript.pdf` (32 pages at the export cutoff). Recomputed primary results, event contrasts, history-length outcomes and crossed-replication records are in `build/tables/`. The source paper, eight complete figures, editable panel layouts and compact numerical inputs are committed. Existing figures suffice to build the paper; no Julia installation or large data download is required for this path.

## Regenerate the figures

Julia **1.12.5** and the committed `Manifest.toml` reproduce the package versions used on the calculation server. The local `DynamicsMetrics` dependency is vendored with its original licence. No GUI, SSH connection or messaging credentials are needed.

```sh
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no tests/runtests.jl
python3 scripts/build.py --redraw
```

`--redraw` renders data panels with CairoMakie, assembles panel labels/layouts with XeLaTeX, and rebuilds the paper. `--assemble` only assembles the committed individual panels and rebuilds the paper. All plot data are compact; forecast trajectories are not downloaded.

## Re-execute the experiments

The private prerelease `reproduction-inputs-v1` contains frozen model weights and states, training prefixes, calibration targets, validation/test physical trajectories, QG coefficients, cohort IDs and numerical anchors. These inputs are kept out of Git history. Authenticate GitHub CLI with access to this private repository, then:

```sh
python3 scripts/fetch_artifacts.py
julia --project=. --startup-file=no -t 4 experiments/main.jl duffing smoke
julia --project=. --startup-file=no -t 4 experiments/main.jl duffing prefix
julia --project=. --startup-file=no -t 4 experiments/main.jl duffing test
```

`smoke` checks the first autonomous forecast against its original SHA-256. `prefix` actually runs both policies for ten base intervals and compares window scores with the original test prefix. `test` runs all 200 base intervals for no correction and the originally selected scheme. Substitute `cdv`, `ks200`, `rp20` or `thomas16` for the other cases. Outputs are separate from immutable inputs, under `runs/`.

Additional commands, using the original settings without new tuning:

```sh
# BO12 on validation only; includes the no-correction candidate in final selection
julia --project=. --startup-file=no -t 4 experiments/main.jl duffing bo
# Training from the saved original training prefix (writes a new model directory)
julia --project=. --startup-file=no -t 4 experiments/main.jl duffing train
# Fixed climatology and finite-300 expected CRPS on the original test truth
julia --project=. --startup-file=no experiments/climatology.jl duffing
# From-equation regeneration of calibration and reference statistics; can be expensive
julia --project=. --startup-file=no experiments/data_and_calibration.jl duffing
# Common-parent history × boundary-input controls, Duffing or KS200
julia --project=. --startup-file=no -t 4 experiments/controls.jl duffing
# Dense postselection history-length sweep, Duffing or KS200 (two threads in paper)
julia --project=. --startup-file=no -t 2 experiments/history.jl duffing
# Three independently trained ensembles × two physical starts; substantial computation
julia --project=. --startup-file=no -t 2 experiments/replication.jl duffing
# Regenerate and check the published individual sequential-model traces
julia --project=. --startup-file=no experiments/sequential_examples.jl lstm
julia --project=. --startup-file=no experiments/sequential_examples.jl esn
# Original homogeneous 300-member validation settings, no architecture ranking
julia --project=. --startup-file=no experiments/sequential.jl lstm
# Optional complete retraining; much more expensive than loading frozen weights
julia --project=. --startup-file=no experiments/sequential.jl lstm --retrain
# Analytical oracle-removal counterexample; no dependencies
python3 experiments/oracle_removal.py
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for result-to-source mappings, numerical guarantees and retention limits, and [VALIDATION.md](docs/VALIDATION.md) for checks actually performed during extraction. Full experiments are deliberately opt-in: reproducing every seed and history length can take days. Recorded wall times depend on hardware and concurrent work and are not exact reproducibility targets.

## Scientific scope

Own candidates and returned populations are distinct records. The controls separate donor boundary input from generated-history replay; output-record copying is not the mechanistic factor. The finite empirical CRPS is used, with an `N²` denominator. Events and adjacent windows are not independent runs. The effective dimension is a correlation diagnostic, not an independent-member count. The repository preserves unfavourable and undefined results and the disclosed Duffing reserve-pool extension.

This is a private working release, not a public code/data deposit or journal submission. Publication licence, persistent DOI and author declarations remain author decisions; see [AUTHOR_ACTIONS.md](docs/AUTHOR_ACTIONS.md).
