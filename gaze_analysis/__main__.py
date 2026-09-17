"""Run analyses directly on the distributed feature CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import chains as c
from . import statistics as s
from . import sensitivity

ROOT = Path(__file__).resolve().parents[1]
FILENAMES = {
    "maptask": "maptask_reference_gaze_features.csv",
    "mundex": "mundex_understanding_gaze_features.csv",
}


def load_tables(data_dir: Path, *, round_trip: bool) -> dict[str, pd.DataFrame]:
    schema = json.loads((data_dir / "feature_schema.json").read_text())
    tables = {}
    for corpus, filename in FILENAMES.items():
        df = pd.read_csv(data_dir / filename, float_precision="round_trip" if round_trip else None)
        expected = {key for group in schema[corpus].values() for key in group}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        if df.empty:
            raise ValueError(f"{filename}: no observations")
        features = [key for group, columns in schema[corpus].items() if group != "metadata" for key in columns]
        if not np.isfinite(df[features].to_numpy(dtype=float)).all():
            raise ValueError(f"{filename}: gaze features must be finite")
        target = "label_aligned" if corpus == "maptask" else "label_understood"
        if set(df[target].unique()) != {0, 1}:
            raise ValueError(f"{filename}: expected both binary target classes")
        tables[corpus] = df
    return tables


def associations(primary, followup, output: Path, report: list[str]) -> None:
    for corpus, target, features, cluster, role in [
        ("maptask", "label_aligned", s.MAPTASK_STRUCTURED_GAZE, "dialogue_id", "speaker"),
        ("mundex", "label_understood", s.MUNDEX_STRUCTURED_GAZE, "interaction_id", "annotator_role"),
    ]:
        mw = c.bh_adjust_within(s.mann_whitney_tests(primary[corpus], target, features))
        mw.to_csv(output / f"{corpus}_mann_whitney.csv", index=False)
        c.role_stratified_tests(followup[corpus], role, target, features).to_csv(
            output / f"{corpus}_role_stratified.csv", index=False
        )
        c.clustered_logistic_regression(followup[corpus], target, features, cluster).to_csv(
            output / f"{corpus}_gee_coefficients.csv", index=False
        )
        strongest = mw.loc[mw.rank_biserial_r.abs().nlargest(4).index]
        report += [f"## {corpus.title()}: largest pooled associations", "",
                   s.markdown_table(strongest[["feature", "rank_biserial_r", "q_value"]],
                                    ["rank_biserial_r", "q_value"]), ""]
    c.condition_stratified_tests(followup["maptask"]).to_csv(output / "maptask_condition_stratified.csv", index=False)
    c.bh_adjust_within(s.kruskal_tests(primary["mundex"], "status", s.MUNDEX_STRUCTURED_GAZE)).to_csv(
        output / "mundex_kruskal.csv", index=False
    )


def prediction(tables, output: Path, report: list[str]) -> None:
    maptask = tables["maptask"].assign(
        is_ec=(tables["maptask"].condition == "ec").astype(int),
        is_giver=(tables["maptask"].speaker == "giver").astype(int),
    )
    mundex = tables["mundex"].assign(role_is_ee=(tables["mundex"].annotator_role == "EE").astype(int))
    for corpus, df, target, group, raw, structured, prefixes, controls, folds in [
        ("maptask", maptask, "label_aligned", "dialogue_id", s.MAPTASK_RAW_GAZE,
         s.MAPTASK_STRUCTURED_GAZE, ("spk_", "addr_"), ["is_ec", "is_giver"], 10),
        ("mundex", mundex, "label_understood", "explainer_id", s.MUNDEX_RAW_GAZE,
         s.MUNDEX_STRUCTURED_GAZE, ("ex_", "ee_"), ["role_is_ee"], 5),
    ]:
        print(f"Fitting {corpus} grouped prediction models...", flush=True)
        models = s.evaluate_binary_models(df, target, group, raw, structured, controls, folds)
        ablation = s.evaluate_feature_ablation(df, target, group, s.ablation_specs(raw, structured, *prefixes), folds)
        models.to_csv(output / f"{corpus}_model_results.csv", index=False)
        ablation.to_csv(output / f"{corpus}_ablation.csv", index=False)
        s.lr_coefficients(df, target, structured).to_csv(output / f"{corpus}_lr_coefficients.csv", index=False)
        report += [f"## {corpus.title()}: grouped prediction", "",
                   s.markdown_table(models[["model", "macro_f1", "cv_splits"]], ["macro_f1"]), "",
                   s.markdown_table(ablation[["feature_set", "n_features", "macro_f1"]], ["macro_f1"]), ""]


def reference_chains(maptask, output: Path, report: list[str]) -> None:
    chains = c.build_reference_chains(maptask)
    c.summarize_chain_positions(chains).to_csv(output / "maptask_chain_position_summary.csv", index=False)
    c.first_vs_remention_tests(chains).to_csv(output / "maptask_first_vs_remention.csv", index=False)
    resolution = c.resolution_transition_tests(chains)
    resolution.to_csv(output / "maptask_resolution_transitions.csv", index=False)
    c.position_matched_tests(chains).to_csv(output / "maptask_position_matched.csv", index=False)
    identities = pd.DataFrame([
        {"dialogue_id": pre.dialogue_id, "concept_id": pre.concept_id,
         "pre_reference_id": pre.reference_id, "post_reference_id": post.reference_id}
        for pre, post in c.resolution_pair_rows(chains)
    ])
    identities.to_csv(output / "maptask_resolution_pairs.csv", index=False)
    report += ["## Same-speaker reference-chain resolution", "",
               f"{len(identities)} pairs in {identities.dialogue_id.nunique()} dialogues.", "",
               s.markdown_table(resolution[["feature", "mean_diff", "q_value"]], ["mean_diff", "q_value"]), ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--analysis", choices=["all", "associations", "prediction", "chains", "sensitivity"], default="all")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.data_dir.resolve():
        parser.error("Choose an output directory separate from data.")
    primary = load_tables(args.data_dir, round_trip=True)
    # The published follow-up workflow read saved CSVs with Pandas' default parser.
    # Preserve that convention for ties in rank/paired tests; primary calculations
    # use round-trip parsing to recover the extracted floating-point values.
    followup = load_tables(args.data_dir, round_trip=False)
    components = (sensitivity.load_components(args.data_dir, followup["maptask"])
                  if args.analysis in {"all", "sensitivity"} else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for corpus, groups in [("maptask", ["status", "condition", "speaker"]),
                           ("mundex", ["status", "annotator_role", "explainer_id"])]:
        s.write_counts(primary[corpus], args.output_dir / f"{corpus}_dataset_counts.csv", groups)
    report = ["# Gaze feature analyses", "",
              f"Analysis selection: {args.analysis}.", "",
              "Positive effects indicate larger values for aligned / UND observations.",
              "Mann–Whitney and paired-test q-values use within-family BH correction;",
              "these are distinct from the clustered GEE results in their own CSVs.", ""]
    if args.analysis in {"all", "associations"}:
        associations(primary, followup, args.output_dir, report)
    if args.analysis in {"all", "prediction"}:
        prediction(primary, args.output_dir, report)
    if args.analysis in {"all", "chains"}:
        reference_chains(followup["maptask"], args.output_dir, report)
    if args.analysis in {"all", "sensitivity"}:
        sensitivity.run(primary, followup, components, args.output_dir / "sensitivity", report)
    (args.output_dir / "summary.md").write_text("\n".join(report) + "\n")
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
