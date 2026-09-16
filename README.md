# Gaze as Evidence for Common Grounding

Processed gaze features and analysis code for **Gaze as Evidence for Common
Grounding: A Cross-Corpus Analysis of MapTask and MUNDEX**.

Nan Li, Albert Gatt and Massimo Poesio · MINT 2026.

## Data

| Corpus | File | Rows × columns | Observation |
| --- | --- | --- | --- |
| MapTask | [Feature table](data/maptask_reference_gaze_features.csv) | 5,144 × 96 | Reference-expression–landmark pair |
| MUNDEX | [Feature table](data/mundex_understanding_gaze_features.csv) | 807 × 94 | Retrospective EX judgment or EE self-report |

These are the feature tables used in the paper. They contain window-level gaze
measurements and annotation metadata. Read the [data guide](data/README.md) for
labels, windows, denominators, dependencies between rows, source attribution
and reuse terms. [Column descriptions](data/feature_schema.json) cover every
column; [file hashes](data/manifest.json) identify this data version.

## Run the analyses

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/chnln/gaze-as-grounding-evidence.git
cd gaze-as-grounding-evidence
uv sync --locked
uv run python -m gaze_analysis
```

The command uses only the files in this repository. It writes 21 CSVs and a
readable `summary.md` under `results/`, which is excluded from Git. Use another
output directory or run one group of analyses with:

```bash
uv run python -m gaze_analysis --output-dir results/my-run
uv run python -m gaze_analysis --analysis associations
uv run python -m gaze_analysis --analysis prediction
uv run python -m gaze_analysis --analysis chains
```

Each run replaces its own output files and summary; unrelated results from a
previous analysis selection remain. Use separate output directories to retain
multiple runs. `--data-dir` accepts another directory containing the same table
filenames and schema.

## Analyses and outputs

| Analysis | Outputs under `results/` |
| --- | --- |
| Dataset composition | `*_dataset_counts.csv` |
| Pooled binary associations (Mann–Whitney, rank-biserial r, BH q) | `*_mann_whitney.csv` |
| MUNDEX four-class associations (Kruskal–Wallis, BH q) | `mundex_kruskal.csv` |
| Role/perspective and eye-contact strata | `*_role_stratified.csv`, `maptask_condition_stratified.csv` |
| Marginal logistic GEE clustered by dialogue/interaction | `*_gee_coefficients.csv` |
| Constant majority, controls and logistic-regression gaze models | `*_model_results.csv` |
| Seven gaze feature-group comparisons | `*_ablation.csv` |
| Full-data standardized structured-feature coefficients | `*_lr_coefficients.csv` |
| Chain position and first versus later mentions | `maptask_chain_position_summary.csv`, `maptask_first_vs_remention.csv` |
| Same-speaker resolution pairs and paired tests | `maptask_resolution_pairs.csv`, `maptask_resolution_transitions.csv` |
| Alignment comparisons within chain-position buckets | `maptask_position_matched.csv` |

Logistic regression standardizes features inside each training fold and uses
balanced class weights. Prediction holds out dialogues in MapTask (10 folds)
and explainers in MUNDEX (5 folds). Scores pool out-of-fold predictions. The
majority row is the constant corpus-majority benchmark. Coefficients fit on all
rows are descriptive fits, separate from held-out performance.

With the pinned environment, the highest feature-group macro-F1 is **.532** for
MapTask (`structured+temporal`) and **.564** for MUNDEX (`raw proportions`). The
chain analysis identifies **189 same-speaker pairs in 45 dialogues**. Effects
and predictive gains are modest; the paper discusses sensitivity to inference
units, repeated participants and annotation perspectives.

## Implementation and numerical conventions

- [statistics.py](gaze_analysis/statistics.py): feature groups, association tests,
  grouped logistic-regression models and descriptive coefficients.
- [chains.py](gaze_analysis/chains.py): role/condition strata, marginal GEE and
  reference-chain construction and comparisons.
- [Command-line runner](gaze_analysis/__main__.py): load the released data and
  write results.

The functions retain the paper's analysis definitions. Main associations and
prediction use round-trip CSV parsing. Role/condition, GEE and chain analyses
retain the original follow-up workflow's default Pandas parsing, which can
matter for ties in rank and paired tests. On Python 3.12 with the locked
packages, all shared fields in the 21 outputs matched the original analysis
references exactly. The public association tables additionally expose BH
q-values; unpublished RF rows are omitted.

This package analyzes the fixed distributed windows. Re-extracting alternative
windows, raw-event timelines and overlap variants requires the original corpus
annotations. Participant-component and annotation-link sensitivity workflows
are outside this package. CR manuscript checking, historical audits and raw
corpora are not required to run the analyses listed above.

## Citation and licenses

Please cite the paper and the upstream datasets; source references are listed
in the [data guide](data/README.md). Machine-readable authorship is provided in
[CITATION.cff](CITATION.cff).

Code: [MIT](LICENSE). Processed data and data documentation:
[CC BY 4.0](data/LICENSE.md), with the upstream attribution notices retained.
