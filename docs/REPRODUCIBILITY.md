# Reproduction contract and evidence inventory

## What is frozen

The main manuscript and appendix correspond to the completed GMD working draft exported on 28 November 2026. The five main comparisons use the validation-selected `(K,m)` recorded in `data/compact/main/<case>/selection.toml`, the same 300-member cohort across policies, and 200 base intervals. The original search is BO12, not an exhaustive 21-point evaluation. No configuration is retuned while packaging this repository.

The fast path reconstructs the complete manuscript from its TeX, bibliography and figure PDFs. The numerical audit recomputes the main table from 200 base-window and 20 distribution-window rows per policy, and the matched-event summaries from event-level contrasts. Appendix settings and archival tables retain their original evidence records. `scripts/prepare_followup_data.py` additionally reconstructs the 56-policy/12-pair plot inputs from their audited source records.

## Figure-to-source map

| Paper item | Inputs and regeneration |
|---|---|
| Figure 1 | `figures/concept/draw.jl`: schematic; no experiment or recovery claim |
| Figure 2 | `data/compact/plot_inputs.jld2`: audited projections of all five completed main tests; `figures/main_panels.jl`. Reference-envelope counts/bounds in `figure2_reference_exceedance_20261125`; `figures/exceedance.jl` |
| Figure 3 | Per-window CSVs in `data/compact/main/<case>/test/`; `figures/main_panels.jl` |
| Figure 4 | Six selected validation paths in `sequence_model_examples_20261124`; `figures/sequential.jl`. `experiments/sequential_examples.jl` independently regenerates those paths from archived model weights and compares every plotted coordinate |
| Figure A1 | Five validation ledgers, selection and scale records; `figures/validation.jl` |
| Figure A2 | `controls/paired_event_contrasts.csv`; `figures/controls.jl` |
| Figure A3 | `followup_results_20261125/history.csv`; `figures/followups.jl`, `experiments/history.jl` |
| Figure A4 | `followup_results_20261125/replication.csv`; `figures/followups.jl`, `experiments/replication.jl` |

Panel letters and layouts are editable TeX. Individual PDFs contain data axes, not baked-in panel letters. `scripts/build.py --redraw` regenerates all eight complete figures. Some PDF metadata/font/library differences are expected; numerical data and axis conventions, rather than byte-identical PDFs, are the portable target.

## Tables and appendix evidence

| Content | Evidence / computation |
|---|---|
| Main presets and training/data settings | `data/compact/main/<case>/{case,config,recorded_data_manifest}.toml`; complete frozen copies in `provenance/input_records` |
| Main W1, spread and CRPS table | `scripts/reproduce_tables.py`, checked against `main_cases_20261123/five_case_summary.csv` |
| Matched 2×2 means and event contrasts | `boundary_history_controls_20261123`, `revision_chaos_pre_20261124/condition_summary.csv`, `controls/paired_event_contrasts.csv`; executable `experiments/controls.jl` |
| Calibration references and availability | `main/<case>/reference.toml`; preserved reference draws in input artifact; from-equation regeneration in `experiments/data_and_calibration.jl` |
| Earlier pilot trade-offs | `state_reconstruction_vs_ensemble_20261122/policy_summaries.csv` and `own_vs_returned_changes.csv`; archival evaluation definitions retained |
| Oracle-removal counterexample | `experiments/oracle_removal.py`, standard library, analytic integrals; writes `build/oracle_removal/` |
| Auxiliary dimension/coverage and own-vs-returned tables | Per-window CSVs in `main/`, `manuscript_review_20261123`, `resync_v4_completed_20261123` |
| Climatology and iid-300 CRPS reference | `manuscript_review_20261123/*_climatology_base.csv` and `climatology_summary.csv`; the fixed calibration marginal and current test truth are preserved in input artifacts |
| Additional earlier-system tests | `resync_v4_completed_20261123`: explicitly archival compact evidence; not rerun under new presets |
| Sequential-model settings and illustration provenance | `configs/sequential_{lstm,esn}.toml`, `sequence_model_examples_20261124/{selected_examples,provenance,...}`; model weights in input artifact |
| History-length and crossed-replication results | Original window summaries/audits in `resync_history_dense_20261124`, `ks_followup_completion_20261127`, `resync_reserve_extension_20261125`; verified reconstruction with `scripts/prepare_followup_data.py` |

Equations and analytical qualifications in the appendix are manuscript source, not numerical simulation outputs. The earlier pilot and auxiliary negative-result tables are preserved numerical records; the package does not invent missing old raw trajectories or claim to regenerate every abandoned development search.

## Model checkpoint portability

The old JLD2 files serialized a complete ESN, including anonymous Julia initializer closures. Those closure names depend on include order. Simply copying the files produced inert reconstructed types in a fresh process.

The exported main checkpoints therefore contain only the original learned parameter tree `ps` and final training state `st`. The loader reconstructs the immutable ESN architecture from the frozen configuration; it does not retrain or modify weights. Initializers are not used by autonomous prediction with loaded parameters and state. `provenance/model_conversion/` maps the original file SHA-256 to the portable file SHA-256. Exact first-forecast hashes and window-level numerical checks validate the conversion. The main scientific update/selection/replay/score formulas are unchanged.

The extraction also removes the parent application's messaging hook and unused graphical backend, makes input paths relative, vendors the local dependency and locks packages to the **actual mac-studio calculation environment**. The laptop's then-current package versions differed and are deliberately not substituted. `provenance/source_export.json` records original source hashes and contiguous extraction ranges; `provenance/files.sha256.json` records final exported files, including explicitly documented portability edits.

## Inputs and role separation

`provenance/artifact.json` gives the archive byte count and SHA-256; `inputs_manifest.json` records each of its files. `scripts/fetch_artifacts.py` verifies the archive before extracting and every file afterwards. It rejects unsafe archive paths and refuses to overwrite an existing input tree.

- Original training prefixes and all 450 main models are preserved.
- Calibration targets, normalization, reference draws, role boundaries and seeds are preserved.
- Separately seeded validation and test physical trajectories are included.
- Test anchors are verification references, not inputs to BO or selection.
- `test` deliberately uses the originally frozen selection; `bo` independently reconstructs validation selection in a new output directory. It does not automatically overwrite the original selection or retune using tests.
- `RESYNC_ENS_INPUTS` and `RESYNC_ENS_RUNS` optionally override `artifacts/inputs` and `runs`. No original-machine path is required.

For a new bank/physical start, use the prescribed replication routine. The three ensembles are crossed with two starts; they are not six independent ensemble banks. The third Duffing bank is trained to a 750-model reserve cap, matching the separately disclosed extension; the originally failed 450-model attempt remains in archival records. KS keeps the original 450 cap. The initial-refill routine and seeds are unchanged.

## Limits and numerical expectations

1. The fast paper build and table audit do not re-execute forecasts. The `test`, `controls`, `history`, `replication`, `data_and_calibration` and sequential commands do.
2. Main production tests and dense sweeps are not all rerun during extraction. The validation record distinguishes exact initialization checks, ten-interval paired runs, complete model example traces and previously audited full-run evidence.
3. Fixed seeds do not guarantee bit-identical chaotic trajectories across arbitrary operating systems, BLAS implementations, Julia or package versions. The locked arm64 macOS environment permits stronger exact-hash checks; archived numeric evidence is always available for comparison. Full-test window checks use absolute tolerance `1e-9`.
4. Older raw forecast arrays were not all retained, notably early CDV pilots. Current main physical inputs and model weights are available to rerun; compact old results remain explicitly archival.
5. Timing includes original hardware/concurrency effects. Recorded single-run wall times are retained for the paper, not promised as reproducible performance benchmarks.
6. A constant-output flag counts flagged members/windows; it is not automatically a physical-failure label. Correlation-based dimension is not independent sample size. Keep undefined values and exclusions.
7. Own and returned records must not be interchanged. An event-wise 2×2 contrast is conditional on the common parent history and does not provide independent-bank uncertainty.

No unpublished hypothesis about a universal history-length threshold is added to the current paper or this scientific evidence package.
