"""Sensitivity analyses on the released feature tables and participant groups.

All resampling settings retain the paper's definitions. Outputs are separated
from the primary tables so alternative inference units remain explicit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.stats.multitest import multipletests

from . import chains as c
from . import statistics as s
from .chains import CHAIN_FEATURES, KEY_SPEAKER_FEATURES, build_reference_chains
from .statistics import mann_whitney_tests

N_BOOT = 800
N_PARTITIONS = 30
WINDOW_KEY = ["dialogue_id", "speaker", "reference_start", "reference_end"]


def load_components(data_dir: Path, maptask: pd.DataFrame) -> dict[str, str]:
    """Validate the dialogue-to-participant-component mapping before analysis."""
    table = pd.read_csv(data_dir / "maptask_participant_components.csv", dtype=str)
    if set(table.columns) != {"dialogue_id", "component"}:
        raise ValueError("Participant mapping needs dialogue_id and component columns.")
    if table.isna().any().any() or table.apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Participant mapping contains empty identifiers.")
    if not table["dialogue_id"].is_unique:
        raise ValueError("Participant mapping must have one row per dialogue.")
    if set(table["dialogue_id"]) != set(maptask["dialogue_id"].astype(str)):
        raise ValueError("Participant mapping must cover exactly the feature-table dialogues.")
    if table["component"].nunique() < 2:
        raise ValueError("Participant-cluster inference needs at least two components.")
    return dict(zip(table["dialogue_id"], table["component"]))


def bh(p: pd.Series) -> np.ndarray:
    return multipletests(p, method="fdr_bh")[1]


def gee_table(df: pd.DataFrame, target: str, features: list[str], group: str, cov_type: str) -> pd.DataFrame:
    """Fit one standardized-feature GEE with the selected covariance correction."""
    rows = []
    y = df[target].astype(int)
    groups = df[group].astype(str)
    for feature in features:
        if df[feature].std() == 0:
            continue
        x = (df[feature] - df[feature].mean()) / df[feature].std()
        fit = sm.GEE(y, sm.add_constant(x.to_frame()), groups=groups, family=Binomial(),
                     cov_struct=Exchangeable()).fit(cov_type=cov_type)
        rows.append({"feature": feature, "coef": float(fit.params[feature]),
                     "se": float(fit.bse[feature]), "p_value": float(fit.pvalues[feature])})
    out = pd.DataFrame(rows)
    out["q_value"] = bh(out["p_value"])
    return out


def mann_whitney(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    out = s.mann_whitney_tests(df, target_col=target, feature_cols=features)
    return out[["feature", "rank_biserial_r", "p_value"]]


def clustering(maptask: pd.DataFrame, mundex: pd.DataFrame, components: dict[str, str], output: Path) -> None:
    maptask = maptask.assign(component=maptask["dialogue_id"].map(components))
    specs = [
        ("MapTask", maptask, "label_aligned", s.MAPTASK_STRUCTURED_GAZE,
         [("dialogue", "dialogue_id"), ("participant group", "component")]),
        ("MUNDEX", mundex, "label_understood", s.MUNDEX_STRUCTURED_GAZE,
         [("interaction", "interaction_id"), ("explainer", "explainer_id")]),
    ]
    tables = []
    for corpus, df, target, features, clusters in specs:
        mw = mann_whitney(df, target, features).rename(
            columns={"rank_biserial_r": "mw_r", "p_value": "mw_p_value"})
        for cluster, column in clusters:
            for cov_type in ("robust", "bias_reduced"):
                t = gee_table(df, target, features, column, cov_type).merge(mw, on="feature", how="left")
                t.insert(0, "cov_type", cov_type)
                t.insert(0, "n_clusters", df[column].nunique())
                t.insert(0, "cluster", cluster)
                t.insert(0, "corpus", corpus)
                tables.append(t)
    pd.concat(tables, ignore_index=True).to_csv(output / "gee_clustering.csv", index=False)


def chain_units(maptask: pd.DataFrame, components: dict[str, str], output: Path) -> None:
    pairs = c.build_resolution_pairs(build_reference_chains(maptask), require_same_speaker=True)
    by_group = {
        "dialogue": pairs,
        "participant group": pairs.assign(dialogue_id=pairs["dialogue_id"].map(components)),
    }
    rows = []
    for unit, frame in by_group.items():
        # The bootstrap resamples the "dialogue_id" column, so relabelling it
        # resamples participant groups with the same seed and replicate count.
        effects = c.bootstrap_paired_effects(frame, N_BOOT).set_index("feature")
        agg_p = {}
        for feature in CHAIN_FEATURES:
            means = frame.groupby("dialogue_id")[f"diff_{feature}"].mean()
            agg_p[feature] = float(stats.wilcoxon(means).pvalue)
        q = dict(zip(agg_p, bh(pd.Series(agg_p))))
        for feature in CHAIN_FEATURES:
            e = effects.loc[feature]
            rows.append({"unit": unit, "feature": feature, "n_pairs": int(e["n_pairs"]),
                         "n_units": frame["dialogue_id"].nunique(),
                         "dz": e["standardized_mean_change"], "ci_low": e["ci_low"], "ci_high": e["ci_high"],
                         "pair_wilcoxon_q": e["wilcoxon_q"],
                         "unit_mean_wilcoxon_p": agg_p[feature], "unit_mean_wilcoxon_q": q[feature]})
    pd.DataFrame(rows).to_csv(output / "chain_clustering.csv", index=False)

    frame = by_group["participant group"].assign(source_dialogue=pairs["dialogue_id"])
    per = frame.groupby("dialogue_id").agg(
        n_dialogues=("source_dialogue", "nunique"), n_pairs=("diff_spk_entropy", "size"),
        mean_diff_spk_entropy=("diff_spk_entropy", "mean"))
    per.index.name = "component"
    per.reset_index().to_csv(output / "chain_by_component.csv", index=False)


def macro_f1(df: pd.DataFrame, target: str, group: str, cols: list[str], cv) -> float:
    y = df[target].astype(int).to_numpy()
    X = df[cols].fillna(0.0).to_numpy()
    pred = cross_val_predict(s.logistic_regression(), X, y, cv=cv, groups=df[group].astype(str).to_numpy())
    return float(f1_score(y, pred, average="macro", zero_division=0))


def cv_partitions(maptask: pd.DataFrame, mundex: pd.DataFrame, output: Path) -> None:
    specs = [
        ("MapTask", maptask.assign(is_ec=(maptask["condition"] == "ec").astype(int),
                                   is_giver=(maptask["speaker"] == "giver").astype(int)),
         "label_aligned", "dialogue_id", 10, ["is_ec", "is_giver"],
         s.ablation_specs(s.MAPTASK_RAW_GAZE, s.MAPTASK_STRUCTURED_GAZE, "spk_", "addr_")),
        ("MUNDEX", mundex.assign(role_is_ee=(mundex["annotator_role"] == "EE").astype(int)),
         "label_understood", "explainer_id", 5, ["role_is_ee"],
         s.ablation_specs(s.MUNDEX_RAW_GAZE, s.MUNDEX_STRUCTURED_GAZE, "ex_", "ee_")),
    ]
    rows = []
    for corpus, df, target, group, splits, controls, ablation in specs:
        sets = [("controls only", controls)] + list(ablation)
        partitions = [("default", GroupKFold(n_splits=splits))] + [
            (str(seed), GroupKFold(n_splits=splits, shuffle=True, random_state=seed))
            for seed in range(N_PARTITIONS)]
        for partition, cv in partitions:
            print(f"  {corpus}: CV partition {partition}", flush=True)
            for name, cols in sets:
                rows.append({"corpus": corpus, "partition": partition, "feature_set": name,
                             "macro_f1": macro_f1(df, target, group, cols, cv)})
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "cv_partitions.csv", index=False)

    summary = []
    for corpus, frame in scores.groupby("corpus"):
        wide = frame.pivot(index="partition", columns="feature_set", values="macro_f1")
        default = wide.loc["default"]
        shuffled = wide.drop(index="default")
        gaze_sets = [c for c in wide.columns if c != "controls only"]
        best = default[gaze_sets].idxmax()
        top_share = (shuffled[gaze_sets].idxmax(axis=1) == best).mean()
        for column in wide.columns:
            values = shuffled[column]
            summary.append({"corpus": corpus, "quantity": column, "default": default[column],
                            "mean": values.mean(), "sd": values.std(), "min": values.min(), "max": values.max()})
        gap = shuffled[best] - shuffled["controls only"]
        summary.append({"corpus": corpus, "quantity": f"{best} minus controls only",
                        "default": default[best] - default["controls only"],
                        "mean": gap.mean(), "sd": gap.std(), "min": gap.min(), "max": gap.max()})
        summary.append({"corpus": corpus, "quantity": f"share of partitions where {best} scores highest",
                        "default": 1.0, "mean": top_share, "sd": np.nan, "min": np.nan, "max": np.nan})
    pd.DataFrame(summary).to_csv(output / "cv_partition_summary.csv", index=False)


def duplicate_windows(maptask: pd.DataFrame, output: Path) -> None:
    grouped = maptask.groupby(WINDOW_KEY)["label_aligned"]
    size = grouped.transform("size")
    duplicated = maptask[size > 1]
    conflicting = duplicated.groupby(WINDOW_KEY)["label_aligned"].nunique().gt(1)
    variants = {
        "all rows": maptask,
        "one row per window, non-aligned if any": maptask.assign(
            label_aligned=grouped.transform("min")).drop_duplicates(WINDOW_KEY),
        "one row per window, conflicting windows dropped": maptask[
            grouped.transform("nunique").eq(1)].drop_duplicates(WINDOW_KEY),
    }
    pd.DataFrame([{
        "rows_sharing_a_window": len(duplicated),
        "shared_windows": duplicated.groupby(WINDOW_KEY).ngroups,
        "shared_windows_with_conflicting_labels": int(conflicting.sum()),
        **{f"rows: {k}": len(v) for k, v in variants.items()},
    }]).to_csv(output / "duplicate_windows_summary.csv", index=False)

    association, prediction = [], []
    ablation = s.ablation_specs(s.MAPTASK_RAW_GAZE, s.MAPTASK_STRUCTURED_GAZE, "spk_", "addr_")
    for name, df in variants.items():
        mw = mann_whitney(df, "label_aligned", s.MAPTASK_STRUCTURED_GAZE)
        gee = gee_table(df, "label_aligned", s.MAPTASK_STRUCTURED_GAZE, "dialogue_id", "robust")
        association.append(mw.merge(gee[["feature", "coef", "q_value"]], on="feature").assign(variant=name))
        df = df.assign(is_ec=(df["condition"] == "ec").astype(int), is_giver=(df["speaker"] == "giver").astype(int))
        for set_name, cols in [("controls only", ["is_ec", "is_giver"])] + list(ablation):
            prediction.append({"variant": name, "feature_set": set_name,
                               "macro_f1": macro_f1(df, "label_aligned", "dialogue_id", cols, GroupKFold(10))})
    pd.concat(association).to_csv(output / "duplicate_windows_association.csv", index=False)
    pd.DataFrame(prediction).to_csv(output / "duplicate_windows_prediction.csv", index=False)

    pairs = c.build_resolution_pairs(build_reference_chains(maptask), require_same_speaker=True)
    shared = set(zip(duplicated["dialogue_id"], duplicated["reference_id"].astype(str)))
    touches = [((d, str(a)) in shared) or ((d, str(b)) in shared)
               for d, a, b in zip(pairs["dialogue_id"], pairs["pre_reference_id"], pairs["post_reference_id"])]
    pd.DataFrame([{"same_speaker_pairs": len(pairs), "pairs_touching_shared_windows": int(sum(touches))}]).to_csv(
        output / "duplicate_windows_chains.csv", index=False)


def interaction_gee(df, target_col, feature_cols, moderator_col, group_col):
    rows = []
    for col in feature_cols:
        sub = df[[target_col, col, moderator_col, group_col]].dropna()
        x = (sub[col] - sub[col].mean()) / sub[col].std()
        m = sub[moderator_col].astype(int)
        X = pd.DataFrame({"const": 1.0, "feature": x, "moderator": m, "interaction": x * m})
        y = sub[target_col].astype(int)
        groups = sub[group_col].astype(str)
        result = sm.GEE(
            y, X, groups=groups,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        rows.append({
            "feature": col,
            "interaction_coef": float(result.params["interaction"]),
            "interaction_p": float(result.pvalues["interaction"]),
        })
    out = pd.DataFrame(rows).sort_values("interaction_p")
    out["interaction_q"] = multipletests(out["interaction_p"], method="fdr_bh")[1]
    out["sig_q05"] = out["interaction_q"] < 0.05
    return out


def role_power_check(
    maptask_df: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample giver rows to the follower sample size (unadjusted p < .05).

    This descriptive rejection-rate check does not preserve dialogue clusters
    and is separate from the formal role-interaction test.
    """
    giver = maptask_df[maptask_df["speaker"] == "giver"].reset_index(drop=True)
    follower = maptask_df[maptask_df["speaker"] == "follower"]
    n_target = len(follower)
    rng = np.random.default_rng(seed)

    hits = {feature: 0 for feature in KEY_SPEAKER_FEATURES}
    for _ in range(n_iter):
        subset = giver.iloc[rng.choice(len(giver), size=n_target, replace=False)]
        positive = subset[subset["label_aligned"] == 1]
        negative = subset[subset["label_aligned"] == 0]
        for feature in KEY_SPEAKER_FEATURES:
            _, p_value = stats.mannwhitneyu(
                positive[feature], negative[feature], alternative="two-sided"
            )
            if p_value < 0.05:
                hits[feature] += 1

    follower_tests = mann_whitney_tests(
        follower, target_col="label_aligned", feature_cols=KEY_SPEAKER_FEATURES
    )
    follower_tests = follower_tests.set_index("feature")
    rows = []
    for feature in KEY_SPEAKER_FEATURES:
        rows.append(
            {
                "feature": feature,
                "giver_n": len(giver),
                "follower_n": n_target,
                "power_giver_at_follower_n": hits[feature] / n_iter,
                "follower_p_value": float(follower_tests.loc[feature, "p_value"]),
                "follower_rank_biserial_r": float(
                    follower_tests.loc[feature, "rank_biserial_r"]
                ),
            }
        )
    return pd.DataFrame(rows)


def majority_row(name: str, subset: pd.DataFrame) -> dict[str, float | int | str]:
    y = subset["label_understood"].astype(int)
    majority = int(y.mean() >= 0.5)
    prediction = pd.Series(majority, index=y.index)
    return {
        "perspective": name,
        "feature_set": "majority baseline",
        "n_features": 0,
        "rows": len(subset),
        "groups": subset["explainer_id"].nunique(),
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
        "f1_positive": f1_score(y, prediction, pos_label=1, zero_division=0),
        "f1_negative": f1_score(y, prediction, pos_label=0, zero_division=0),
    }


def perspective_prediction(mundex: pd.DataFrame, output: Path) -> None:
    """Repeat the fixed five-fold explainer-held-out comparison per perspective."""
    slices = {"pooled": mundex,
              "EX": mundex[mundex["annotator_role"] == "EX"].copy(),
              "EE": mundex[mundex["annotator_role"] == "EE"].copy()}
    if len(slices["EX"]) + len(slices["EE"]) != len(mundex):
        raise ValueError("Unexpected annotator_role outside EX/EE")
    specs = s.ablation_specs(s.MUNDEX_RAW_GAZE, s.MUNDEX_STRUCTURED_GAZE, "ex_", "ee_")
    count_rows, prediction_rows = [], []
    for name, subset in slices.items():
        count_rows.append({
            "perspective": name, "rows": len(subset),
            "interactions": subset["interaction_id"].nunique(),
            "explainers": subset["explainer_id"].nunique(),
            "understood": int(subset["label_understood"].sum()),
            "non_understood": int((1 - subset["label_understood"]).sum()),
            "understood_rate": subset["label_understood"].mean(),
        })
        prediction_rows.append(majority_row(name, subset))
        ablation = s.evaluate_feature_ablation(
            subset, "label_understood", "explainer_id", specs, 5)
        ablation.insert(0, "perspective", name)
        ablation.insert(2, "rows", len(subset))
        ablation.insert(3, "groups", subset["explainer_id"].nunique())
        prediction_rows.extend(ablation.to_dict("records"))
        if name == "pooled":
            pooled = subset.assign(role_is_ee=(subset["annotator_role"] == "EE").astype(int))
            controls = s.evaluate_binary_models(
                pooled, "label_understood", "explainer_id", s.MUNDEX_RAW_GAZE,
                s.MUNDEX_STRUCTURED_GAZE, ["role_is_ee"], 5)
            control = controls[controls["model"] == "LR (controls only)"].iloc[0]
            prediction_rows.append({
                "perspective": name, "feature_set": "controls only", "n_features": 1,
                "rows": int(control["rows"]), "groups": int(control["groups"]),
                **{key: control[key] for key in ["accuracy", "macro_f1", "f1_positive", "f1_negative"]},
            })
    pd.DataFrame(count_rows).to_csv(output / "mundex_perspective_counts.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "mundex_perspective_prediction.csv", index=False)


def supplementary(maptask: pd.DataFrame, mundex: pd.DataFrame, output: Path) -> None:
    """Formal moderator tests, row subsampling, and chain effect sizes."""
    speaker_features = ["spk_prop_partner", "spk_prop_task", "spk_entropy", "spk_transitions"]
    m = maptask.assign(is_giver=(maptask["speaker"] == "giver").astype(int))
    interaction_gee(m, "label_aligned", speaker_features, "is_giver", "dialogue_id").to_csv(
        output / "maptask_role_interaction_gee.csv", index=False)
    m["condition_ec"] = m["dialogue_id"].str.extract(r"q\d+(ec|nc)\d+")[0].eq("ec").astype(int)
    interaction_gee(m, "label_aligned", speaker_features, "condition_ec", "dialogue_id").to_csv(
        output / "maptask_condition_interaction_gee.csv", index=False)
    features = ["ex_prop_task", "ex_prop_partner", "ee_prop_task", "ee_prop_partner",
                "mutual_gaze_explicit", "mutual_gaze_derived"]
    x = mundex.assign(is_ex=(mundex["annotator_role"] == "EX").astype(int))
    interaction_gee(x, "label_understood", features, "is_ex", "interaction_id").to_csv(
        output / "mundex_role_interaction_gee.csv", index=False)
    role_power_check(maptask).to_csv(output / "maptask_role_power_check.csv", index=False)
    chains = build_reference_chains(maptask)
    mixed = c.resolution_transition_tests(chains, require_same_speaker=False).rename(
        columns={"pre_resolution_mean": "pre_mean", "post_resolution_mean": "post_mean"})
    mixed["significant_q05"] = mixed["q_value"] < .05
    mixed[["feature", "n_pairs", "pre_mean", "post_mean", "mean_diff", "p_value", "q_value",
           "significant_q05"]].to_csv(output / "chain_mixed_speaker_results.csv", index=False)
    pairs = c.build_resolution_pairs(chains)
    pairs.to_csv(output / "resolution_pair_changes.csv", index=False)
    c.bootstrap_paired_effects(pairs, N_BOOT).to_csv(output / "resolution_paired_effects.csv", index=False)


def run(primary, followup, components: dict[str, str], output: Path, report: list[str]) -> None:
    """Run the paper's fixed sensitivity comparisons; use no raw annotations."""
    output.mkdir(parents=True, exist_ok=True)
    maptask, mundex = followup["maptask"], followup["mundex"]
    print("Sensitivity: alternative GEE clusters and covariance corrections...", flush=True)
    clustering(maptask, mundex, components, output)
    print("Sensitivity: chain inference units and cluster bootstraps...", flush=True)
    chain_units(maptask, components, output)
    print("Sensitivity: duplicate-window policies...", flush=True)
    duplicate_windows(maptask, output)
    print("Sensitivity: moderator tests, role subsampling, and Figure 2 statistics...", flush=True)
    supplementary(maptask, mundex, output)
    print("Sensitivity: MUNDEX perspective-specific prediction...", flush=True)
    perspective_prediction(primary["mundex"], output)
    print("Sensitivity: default and 30 reshuffled grouped partitions...", flush=True)
    cv_partitions(maptask, mundex, output)
    append_summary(output, report)


def append_summary(output: Path, report: list[str]) -> None:
    """Summarize saved sensitivity results without rounding a small loss to zero."""
    cv_summary = pd.read_csv(output / "cv_partition_summary.csv")
    chain_summary = pd.read_csv(output / "chain_clustering.csv")
    gains = cv_summary[cv_summary.quantity.str.contains("minus controls")].copy()
    for column in ["default", "mean", "sd", "min", "max"]:
        gains[column] = gains[column].map(lambda value: f"{value:.6f}")
    report += ["## Sensitivity analyses", "",
               "Results are in `sensitivity/`. Grouped partitions use seeds 0–29;",
               "cluster bootstraps use 800 replicates and seed 4200; giver-row",
               "subsampling uses 1,000 draws and seed 42.", "",
               "### Prediction gains over controls", "",
               s.markdown_table(gains, []), "",
               "### Speaker entropy under alternative inference units", "",
               s.markdown_table(chain_summary[chain_summary.feature == "spk_entropy"],
                                ["dz", "ci_low", "ci_high", "pair_wilcoxon_q",
                                 "unit_mean_wilcoxon_p", "unit_mean_wilcoxon_q"]), "",
               "Linked-annotation reconstruction, overlap-policy re-extraction and",
               "alternative windows require source annotations and are not run here.", ""]
