# Extraction validation

Checks are run against the isolated `resync-ens` tree, with no source-code import from the parent EESN repository.

Completed on mac-studio with Julia 1.12.5 and the original locked package versions:

- Finite empirical metrics and protected selection: 33/33 tests.
- Fixed-window streaming evaluator: 12/12 tests.
- Matched calibration/exact pooled references: 9/9 tests.
- BO failure handling/reproducibility: 10/10 tests.
- Sequential-model gradients and history alignment: 57/57 tests.
- First autonomous forecast: exact original SHA-256 for Duffing, CDV, KS200, QG20 and cyclic Thomas16, using the exported portable checkpoints.
- Selected LSTM and reservoir illustrations: all three 1,000-step conditions independently recomputed; every plotted physical coordinate agrees to `1e-10`.
- All six data-rendering scripts and the concept renderer execute using committed compact data.

Completed locally:

- Forty primary table values independently reaggregated from per-window CSVs, tolerance `1e-12`.
- Full 56-policy history and 12 crossed-pair input reconstruction/audit.
- Oracle-removal analytical assertions.
- All eight TeX figure assemblies and 32-page manuscript build.

Ten-base-interval paired reruns are complete for all five systems, for both no correction and Resync (10 policy runs). Every checked own/returned spread, spread-loss, CRPS and distribution-window W1 matches its original prefix within `1e-9`; exact differences and first-forecast checks are recorded under `provenance/validation/prefix/`. These are prefix checks, not new 200-interval experiments or new parameter selection. The complete production-run summaries retain their original audit records.
