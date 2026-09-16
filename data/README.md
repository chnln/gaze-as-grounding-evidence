# Data and measurement guide

The CSVs in this directory contain the fixed feature tables used in the MINT
2026 paper. These are window-level features derived from gaze annotations,
not raw eye-tracker samples or complete gaze-event tracks. Column descriptions
are in [feature_schema.json](feature_schema.json), and SHA-256 hashes are in
[manifest.json](manifest.json).

## Observations, labels, and roles

| Corpus | Observation | Positive / negative target | Gaze prefixes |
| --- | --- | --- | --- |
| MapTask | RE–landmark pair | `label_aligned`: aligned=1; pending or misunderstood=0 | `spk_`: producer of this RE; `addr_`: addressee |
| MUNDEX | Retrospective annotation | `label_understood`: UND=1; PART_UND, NON_UND, MISUND=0 | `ex_`: explainer; `ee_`: explainee |

MapTask has 5,144 rows and 96 columns. Multiple landmarks in one expression can
produce identical gaze windows: 54 rows across 26 windows, four of which have
conflicting binary labels. Concept-level rows are retained. The paper reports both duplicate-window
sensitivity variants in Appendix A.1.

MUNDEX has 807 rows and 94 columns. `annotator_role` identifies whose judgment
supplied the target; it does not change the meaning of `ex_`/`ee_` gaze columns.
EX judges the explainee and EE self-reports. These are pooled judgments, not
consensus or objective comprehension labels. The 956 valid judgments in
26 gaze-complete interactions yield 807 windows after coverage filtering
(458 EX, 349 EE). Appendix A.3 reports linking and perspective sensitivities.

## Labels, windows, and denominators

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

## Extended features

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

## Feature sets and analysis interpretation

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

Prediction holds out dialogues in MapTask (10 folds) and explainers in MUNDEX
(5 folds). Dialogue-held-out MapTask evaluation can still share participants
between training and test sets; it does not estimate transfer to entirely unseen
participants. Linked MUNDEX perspectives remain distinct judgments, not
independent event-level consensus labels.

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
MUNDEX v0.7 specify CC BY 4.0 in their release materials. See [data/LICENSE.md](LICENSE.md)
for the data terms and retained copyright attribution.
