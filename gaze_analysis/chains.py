"""Role, condition, clustered-association and reference-chain analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .statistics import MAPTASK_STRUCTURED_GAZE, MUNDEX_STRUCTURED_GAZE, mann_whitney_tests


CHAIN_FEATURES = [
    "spk_prop_partner",
    "spk_prop_task",
    "spk_entropy",
    "spk_transitions",
    "addr_prop_partner",
    "addr_prop_task",
    "addr_entropy",
    "mutual_gaze",
]


KEY_SPEAKER_FEATURES = [
    "spk_prop_task",
    "spk_prop_partner",
    "spk_entropy",
    "spk_transitions",
]


def bh_adjust_within(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Add BH/FDR-adjusted q-values, within each group if group_col is given."""
    df = df.copy()
    if group_col is None:
        df["q_value"] = multipletests(df["p_value"], method="fdr_bh")[1]
    else:
        df["q_value"] = np.nan
        for _, index in df.groupby(group_col).groups.items():
            df.loc[index, "q_value"] = multipletests(
                df.loc[index, "p_value"], method="fdr_bh"
            )[1]
    df["significant_q_lt_0_05"] = df["q_value"] < 0.05
    return df


def role_stratified_tests(
    df: pd.DataFrame,
    role_col: str,
    target_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    frames = []
    for role, subset in df.groupby(role_col):
        if subset[target_col].nunique() < 2:
            continue
        cols = [col for col in dict.fromkeys(feature_cols) if col in subset.columns]
        result = mann_whitney_tests(subset, target_col=target_col, feature_cols=cols)
        result.insert(0, "role", role)
        result.insert(1, "n_rows", len(subset))
        frames.append(result)
    return bh_adjust_within(pd.concat(frames, ignore_index=True), group_col="role")


def condition_stratified_tests(maptask_df: pd.DataFrame) -> pd.DataFrame:
    """Aligned vs. non-aligned tests within the eye-contact (ec) and
    no-eye-contact (nc) conditions, encoded in the MapTask dialogue ID."""
    df = maptask_df.copy()
    df["condition"] = df["dialogue_id"].str.extract(r"q\d+(ec|nc)\d+")[0]
    if df["condition"].isna().any():
        raise ValueError("Unparsable MapTask dialogue IDs for condition split.")
    result = role_stratified_tests(
        df,
        role_col="condition",
        target_col="label_aligned",
        feature_cols=MAPTASK_STRUCTURED_GAZE + ["mutual_gaze"],
    )
    return result.rename(columns={"role": "condition"})


def position_matched_tests(chains: pd.DataFrame) -> pd.DataFrame:
    """Aligned vs. non-aligned within each mention-position bucket."""
    frames = []
    for bucket, subset in chains.groupby("mention_bucket"):
        if subset["label_aligned"].nunique() < 2:
            continue
        result = mann_whitney_tests(
            subset, target_col="label_aligned", feature_cols=KEY_SPEAKER_FEATURES
        )
        result.insert(0, "mention_bucket", bucket)
        result.insert(1, "n_rows", len(subset))
        frames.append(result)
    return bh_adjust_within(
        pd.concat(frames, ignore_index=True), group_col="mention_bucket"
    )


def clustered_logistic_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    group_col: str,
) -> pd.DataFrame:
    """Per-feature logistic GEE with exchangeable within-group correlation.

    One univariate model per feature, mirroring the marginal Mann-Whitney
    tests: this asks whether each marginal association survives
    dialogue/interaction-level clustering. A joint multivariate model is
    deliberately avoided here because the structured features are strongly
    collinear (proportions sum to coverage; entropy tracks transitions),
    which makes joint coefficient signs uninterpretable."""
    y = df[target_col].astype(int)
    groups = df[group_col].astype(str)
    rows = []
    for col in feature_cols:
        if col not in df.columns or df[col].std() == 0:
            continue
        x = (df[col] - df[col].mean()) / df[col].std()
        X = sm.add_constant(x.to_frame())
        result = sm.GEE(
            y,
            X,
            groups=groups,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        rows.append(
            {
                "feature": col,
                "coef_log_odds_per_sd": float(result.params[col]),
                "cluster_robust_se": float(result.bse[col]),
                "p_value": float(result.pvalues[col]),
            }
        )
    return bh_adjust_within(pd.DataFrame(rows).sort_values("p_value"))


def build_reference_chains(maptask_df: pd.DataFrame) -> pd.DataFrame:
    """Order mentions of the same landmark concept within a dialogue."""
    df = maptask_df.sort_values(["dialogue_id", "concept_id", "reference_start"]).copy()
    df["mention_index"] = df.groupby(["dialogue_id", "concept_id"]).cumcount() + 1
    df["chain_length"] = df.groupby(["dialogue_id", "concept_id"])["mention_index"].transform("max")
    df["mention_bucket"] = df["mention_index"].clip(upper=4).astype(int).astype(str)
    df.loc[df["mention_index"] >= 4, "mention_bucket"] = "4+"
    return df


def summarize_chain_positions(chains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, subset in chains.groupby("mention_bucket"):
        row: dict[str, float | int | str] = {
            "mention_bucket": bucket,
            "n_mentions": len(subset),
            "aligned_rate": float(subset["label_aligned"].mean()),
        }
        for feature in CHAIN_FEATURES:
            row[f"mean_{feature}"] = float(subset[feature].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mention_bucket")


def first_vs_remention_tests(chains: pd.DataFrame) -> pd.DataFrame:
    chains = chains.copy()
    chains["is_first_mention"] = (chains["mention_index"] == 1).astype(int)
    result = mann_whitney_tests(
        chains,
        target_col="is_first_mention",
        feature_cols=CHAIN_FEATURES + ["label_aligned"],
    )
    result = result.rename(
        columns={"positive_mean": "first_mention_mean", "negative_mean": "remention_mean"}
    )
    return bh_adjust_within(result)


def resolution_pair_rows(
    chains: pd.DataFrame, *, require_same_speaker: bool = True,
):
    """Yield the last non-aligned and resolving aligned row of each chain.

    Pair once per dialogue/concept, then apply the same-speaker restriction.
    Do not search for a later same-speaker resolution after rejecting a pair.
    """
    for _, chain in chains.groupby(["dialogue_id", "concept_id"]):
        chain = chain.sort_values("mention_index")
        statuses = chain["label_aligned"].to_numpy()
        non_aligned_idx = np.where(statuses == 0)[0]
        if len(non_aligned_idx) == 0:
            continue
        first_non_aligned = non_aligned_idx[0]
        later_aligned = np.where(statuses[first_non_aligned + 1 :] == 1)[0]
        if len(later_aligned) == 0:
            continue
        resolve_pos = first_non_aligned + 1 + later_aligned[0]
        # last non-aligned mention before the resolving aligned mention
        pre_pos = max(idx for idx in non_aligned_idx if idx < resolve_pos)
        pre_row = chain.iloc[pre_pos]
        post_row = chain.iloc[resolve_pos]
        if require_same_speaker and pre_row["speaker"] != post_row["speaker"]:
            continue
        yield pre_row, post_row


def resolution_transition_tests(
    chains: pd.DataFrame, *, require_same_speaker: bool = True,
) -> pd.DataFrame:
    """Paired comparison at the last non-aligned and resolving aligned mention."""
    pairs = list(resolution_pair_rows(chains, require_same_speaker=require_same_speaker))
    pre_rows = [pre for pre, _ in pairs]
    post_rows = [post for _, post in pairs]

    pre = pd.DataFrame(pre_rows)
    post = pd.DataFrame(post_rows)
    rows = []
    for feature in CHAIN_FEATURES:
        diffs = post[feature].to_numpy() - pre[feature].to_numpy()
        if np.allclose(diffs, 0):
            continue
        stat, p_value = stats.wilcoxon(diffs)
        rows.append(
            {
                "feature": feature,
                "n_pairs": len(diffs),
                "pre_resolution_mean": float(pre[feature].mean()),
                "post_resolution_mean": float(post[feature].mean()),
                "mean_diff": float(diffs.mean()),
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
                "significant_p_lt_0_05": bool(p_value < 0.05),
            }
        )
    result = pd.DataFrame(rows).sort_values("p_value")
    return bh_adjust_within(result)



def build_resolution_pairs(chains: pd.DataFrame, *, require_same_speaker: bool = True) -> pd.DataFrame:
    features = [
        "spk_prop_task",
        "spk_prop_partner",
        "spk_entropy",
        "spk_transitions",
        "addr_prop_task",
        "addr_prop_partner",
        "addr_entropy",
        "mutual_gaze",
    ]
    rows = []
    for pre, post in resolution_pair_rows(chains, require_same_speaker=require_same_speaker):
        row: dict[str, object] = {
            "dialogue_id": pre["dialogue_id"],
            "concept_id": pre["concept_id"],
            "pre_reference_id": pre["reference_id"],
            "post_reference_id": post["reference_id"],
        }
        for feature in features:
            row[f"pre_{feature}"] = pre[feature]
            row[f"post_{feature}"] = post[feature]
            row[f"diff_{feature}"] = post[feature] - pre[feature]
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_paired_effects(pairs: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    features = [col.removeprefix("diff_") for col in pairs.columns if col.startswith("diff_")]
    groups = {name: idx.to_numpy() for name, idx in pairs.groupby("dialogue_id").groups.items()}
    keys = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(4200)
    rows = []
    for feature in features:
        diff = pairs[f"diff_{feature}"].to_numpy()
        point = float(diff.mean() / diff.std(ddof=1))
        boot = []
        for _ in range(n_boot):
            selected = rng.choice(keys, size=len(keys), replace=True)
            index = np.concatenate([groups[key] for key in selected])
            values = pairs.loc[index, f"diff_{feature}"].to_numpy()
            sd = values.std(ddof=1)
            if sd > 0:
                boot.append(float(values.mean() / sd))
        low, high = np.quantile(boot, [0.025, 0.975])
        p_value = float(stats.wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else 1.0
        rows.append(
            {
                "feature": feature,
                "standardized_mean_change": point,
                "ci_low": float(low),
                "ci_high": float(high),
                "wilcoxon_p": p_value,
                "n_pairs": len(diff),
            }
        )
    df = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    _, q_values, _, _ = multipletests(df["wilcoxon_p"].values, method="fdr_bh")
    df["wilcoxon_q"] = q_values
    return df
