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
measurements and annotation metadata derived from gaze annotations, not raw
eye-tracker samples or complete gaze-event tracks. [Column descriptions](data/feature_schema.json)
cover every column; [file hashes](data/manifest.json) identify this data version.

- [Run the analyses](#run-the-analyses)
- [Analyses and outputs](#analyses-and-outputs)
- [Data and measurement definitions](#data-and-measurement-definitions)
- [Implementation and numerical conventions](#implementation-and-numerical-conventions)
- [Sources and attribution](#sources-and-attribution)
- [Citation and licenses](#citation-and-licenses)

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

## Data and measurement definitions

### Observations, labels, and roles

| Corpus | Observation | Positive / negative target | Gaze prefixes |
| --- | --- | --- | --- |
| MapTask | RE–landmark pair | `label_aligned`: aligned=1; pending or misunderstood=0 | `spk_`: producer of this RE; `addr_`: addressee |
| MUNDEX | Retrospective annotation | `label_understood`: UND=1; PART_UND, NON_UND, MISUND=0 | `ex_`: explainer; `ee_`: explainee |

Multiple landmarks in one MapTask expression can produce identical gaze windows:
54 rows across 26 windows, four of which have conflicting binary labels. Concept-level rows are retained. The paper reports
both duplicate-window sensitivity variants in Appendix A.1.

In MUNDEX, `annotator_role` identifies whose judgment supplied the target; it does
not change the meaning of `ex_`/`ee_` gaze columns.
EX judges the explainee and EE self-reports. These are pooled judgments, not
consensus or objective comprehension labels. The 956 valid judgments in
26 gaze-complete interactions yield 807 windows after coverage filtering
(458 EX, 349 EE). Appendix A.3 reports linking and perspective sensitivities.

### Labels, windows, and denominators

- The shared categories are partner, task, away. MapTask maps up/down/off;
  MUNDEX maps participant-directed gaze/TABLE/AWAY. MapTask up→partner is a
  literal partner-directed interpretation only in the eye-contact condition.
- MapTask windows span the RE plus 1.5 seconds after it. MUNDEX windows expand
  each annotation by 2 seconds on both sides, clipping the start at zero.
- Each participant must have at least 30% observed gaze coverage. Unobserved
  time is not interpolated or treated as away.
- `prop_*` uses **full window duration** as denominator; the three proportions
  sum to `coverage`, not necessarily one.
- `entropy` uses the duration distribution normalized by **observed time**,
  in bits, with maximum log2(3). It measures category diversity, not switching.
- `transitions` counts changes between contributing interval labels, including
  across missing gaps. It is not divided by duration.
- `mutual_gaze` / `mutual_gaze_derived` uses the duration of overlapping unioned
  partner intervals divided by full window duration. MUNDEX explicit mutual
  gaze sums clipped annotated intervals without unioning them.

### Extended features

| Group | Additional columns | Definition |
| --- | ---:| --- |
| Temporal | 42 | 21 per participant: run counts, mean/max durations, switch rate, latencies, and dominant/first/last categories |
| Bigrams | 12 | Six directed category-change proportions per participant; all zero if there are no changes |
| Coordination | 6 | Sampled joint-category proportions, joint entropy, and partner-indicator correlation |
| Ratios | 9 | Participant partner/task ratios, engagement, task dominance, and three absolute asymmetries |

Saved `n_fixations_*`, `mean_fix_dur_*`, and `max_fix_dur_*` names refer to
annotation-derived **gaze runs**, not eye-tracker fixation detections. Same-label
runs merge across gaps of at most .01 seconds, including that gap. The temporal
switch rate divides run-label changes by full window duration. Latencies are
fractions of the window; an absent target has latency one. Dominant-category
ties use partner, task, away order; saved one-hot column names are unchanged.

Coordination uses approximately 10 Hz, with at least 20 grid points. Only
**jointly observed samples** enter its denominator. Thus `mutual_partner` and
`mutual_gaze` have different denominators. Mutual task gaze means a shared
category, not proven attention to the same object. Partner coupling is zero
when either partner indicator is constant.

Ratios use `partner / max(task, .02)`, not an additive epsilon. Engagement is
partner+task; task dominance is task−partner; asymmetries are absolute
between-participant differences.

Cross-label overlaps use different rules for duration features, runs, joint
sampling, and mutual intervals. These are the CR definitions (Appendix B.8).
The earlier-event temporal/bigram sensitivity in Appendix A.4 is separate from
the primary extraction. Unifying policies would define a new dataset version.

### Feature sets and analysis interpretation

| Feature set | MapTask | MUNDEX |
| --- | ---:| ---:|
| Raw proportions | 7 | 8 |
| Structured | 13 | 14 |
| Structured + temporal | 55 | 56 |
| Structured + bigrams | 25 | 26 |
| Structured + coordination | 19 | 20 |
| Structured + ratios | 22 | 23 |
| All extended | 82 | 83 |

Metadata accounts for the remaining 14/11 columns. Controls are separate from
the gaze feature groups. All extended features are defined in the fixed tables.

Association `r` is positive when the positive class tends to have larger values;
mean differences are positive-class minus negative-class means. Distinguish
unadjusted Mann–Whitney p-values, within-family BH q-values, and clustered GEE
q-values: passing one test does not establish robustness under the others.
The paper reports small, dependence-sensitive associations. Clustered GEE
outputs and window-level rank tests answer different inferential questions.

Dialogue-held-out MapTask evaluation can still share participants between
training and test sets; it does not estimate transfer to entirely unseen
participants. Linked MUNDEX perspectives remain distinct judgments, not
independent event-level consensus labels.

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

## Sources and attribution

| Source | Materials used | Version and source |
| --- | --- | --- |
| HCRC Map Task Corpus; Human Communication Research Centre, University of Edinburgh and University of Glasgow | Gaze, timing and landmark-reference annotations | [NXT annotations 2.1](https://groups.inf.ed.ac.uk/maptask/maptasknxt.html) |
| Grounded Misunderstandings in MapTask (GMMT); Nan Li, Albert Gatt and Massimo Poesio | Perspectivist grounding labels | [GMMT repository](https://github.com/chnln/grounded-misunderstandings-in-maptask) |
| MUNDEX Annotations; Hendrik Buschmeier, Angela Grimminger, Petra Wagner, Stefan Lazarov, Olcay Türk and Yu Wang | Gaze and retrospective understanding annotations | [Version 0.7, Zenodo](https://doi.org/10.5281/zenodo.17129817) |

The released tables are derived materials: gaze labels were mapped to shared
categories, clipped to analysis windows, filtered by coverage, and converted
to the measurements described above. Grounding/understanding annotations supply
the targets. Corpus recordings, maps, full transcripts and original annotation
archives are not included.

The HCRC download page identifies CC BY 4.0 as the license for its downloads;
the older annotation archive retains an earlier license notice. GMMT and
MUNDEX v0.7 specify CC BY 4.0 in their release materials. See [data license](data/LICENSE.md)
for the data terms and retained copyright attribution.

## Citation and licenses

Please cite the paper and the upstream datasets; source references are listed
[above](#sources-and-attribution). Machine-readable authorship is provided in
[CITATION.cff](CITATION.cff).

Code: [MIT](LICENSE). Processed data and data documentation:
[CC BY 4.0](data/LICENSE.md), with the upstream attribution notices retained.
