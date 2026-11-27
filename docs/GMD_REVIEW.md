# GMD terminology and scope review — 28 November 2026

The code and input release are public. The manuscript's Code and data availability section links both. An archival DOI remains pending; the authors' licence and declarations are separate submission items.

## Controlled terms

| Term | Meaning |
|---|---|
| Resynchronization (resync) | Complete generated-history intervention and autonomous continuation |
| History replay | Process donor-generated inputs through the recipient's fixed update |
| State reconstruction | Recipient-specific state response to that replay; not a claim of equal hidden states or guaranteed recovery |
| Donor boundary input | Donor's final output used once to resume autonomous prediction |
| Own outputs | Candidate generated before replacement in the current segment |
| Returned outputs | Issued population after donor replacement; denoted `eff` in equations |
| Warm-up initialization | Input-driven state initialization before the forecast |
| Loss washout | Initial samples omitted from a training loss; distinct from forecast initialization |

The manuscript consolidates synonyms for the two records and the two intervention components. It retains resync as the method name; decomposed controls explain the mechanism rather than replacing the main narrative.

## Related language in the literature

Input-driven initialization is established; it is not claimed as new. [Grigoryeva et al. (2024)](https://doi.org/10.1016/j.physd.2024.134325) learn a starting map from short observed histories. Here a donor-generated history is processed through the fixed recipient update. [Du and Smith (2017)](https://doi.org/10.1016/j.physd.2017.06.001) exchange model-generated information in observation space and use data assimilation to reconstruct model states. The distinction made here concerns the implemented fixed-map replay and its own/returned ensemble evaluation, not priority for information exchange or history-based initialization. This targeted comparison is not an exhaustive proof of novelty.

## Prose and evidence

The GMD abstract is shortened from157 to134 words. Discussion and conclusion reduce repeated qualifications while retaining the one-bank primary-test scope, nonuniform CRPS gains, selected LSTM illustration, donor-boundary/history interaction and raw-retention limits. Reservoir synchronization theory is no longer cited as a guarantee for LSTM recovery. No experimental result, parameter, table value or plot data is changed by this edit.
