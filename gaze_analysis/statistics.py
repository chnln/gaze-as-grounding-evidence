"""Feature groups, association tests and grouped logistic-regression analyses."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MAPTASK_RAW_GAZE = [
    "spk_prop_partner",
    "spk_prop_task",
    "spk_prop_away",
    "addr_prop_partner",
    "addr_prop_task",
    "addr_prop_away",
    "mutual_gaze",
]


MAPTASK_STRUCTURED_GAZE = MAPTASK_RAW_GAZE + [
    "spk_coverage",
    "spk_transitions",
    "spk_entropy",
    "addr_coverage",
    "addr_transitions",
    "addr_entropy",
]


MUNDEX_RAW_GAZE = [
    "ex_prop_partner",
    "ex_prop_task",
    "ex_prop_away",
    "ee_prop_partner",
    "ee_prop_task",
    "ee_prop_away",
    "mutual_gaze_derived",
    "mutual_gaze_explicit",
]


MUNDEX_STRUCTURED_GAZE = MUNDEX_RAW_GAZE + [
    "ex_coverage",
    "ex_transitions",
    "ex_entropy",
    "ee_coverage",
    "ee_transitions",
    "ee_entropy",
]


GAZE_LABELS = ("partner", "task", "away")


COORDINATION_COLS = [
    "mutual_task",
    "mutual_partner",
    "gaze_alignment",
    "complementary_gaze",
    "joint_entropy",
    "partner_coupling",
]


def temporal_feature_cols(prefix: str) -> list[str]:
    cols: list[str] = []
    for label in GAZE_LABELS:
        cols += [
            f"{prefix}n_fixations_{label}",
            f"{prefix}mean_fix_dur_{label}",
            f"{prefix}max_fix_dur_{label}",
        ]
    cols += [f"{prefix}switch_rate", f"{prefix}latency_partner", f"{prefix}latency_task"]
    for group in ("dominant_is", "first_is", "last_is"):
        cols += [f"{prefix}{group}_{label}" for label in GAZE_LABELS]
    return cols


def bigram_feature_cols(prefix: str) -> list[str]:
    return [
        f"{prefix}bigram_{a}_{b}" for a in GAZE_LABELS for b in GAZE_LABELS if a != b
    ]


def ratio_feature_cols(prefix_a: str, prefix_b: str) -> list[str]:
    cols: list[str] = []
    for prefix in (prefix_a, prefix_b):
        cols += [
            f"{prefix}partner_task_ratio",
            f"{prefix}engagement",
            f"{prefix}task_dominance",
        ]
    cols += ["partner_asymmetry", "task_asymmetry", "entropy_asymmetry"]
    return cols


def ablation_specs(
    raw_cols: list[str],
    structured_cols: list[str],
    prefix_a: str,
    prefix_b: str,
) -> list[tuple[str, list[str]]]:
    temporal = temporal_feature_cols(prefix_a) + temporal_feature_cols(prefix_b)
    bigrams = bigram_feature_cols(prefix_a) + bigram_feature_cols(prefix_b)
    ratios = ratio_feature_cols(prefix_a, prefix_b)
    return [
        ("raw proportions", raw_cols),
        ("structured", structured_cols),
        ("structured+temporal", structured_cols + temporal),
        ("structured+bigrams", structured_cols + bigrams),
        ("structured+coordination", structured_cols + COORDINATION_COLS),
        ("structured+ratios", structured_cols + ratios),
        ("all extended", structured_cols + temporal + bigrams + COORDINATION_COLS + ratios),
    ]


def write_counts(df: pd.DataFrame, path: Path, group_cols: list[str]) -> None:
    rows = []
    for col in group_cols:
        for value, count in df[col].value_counts(dropna=False).items():
            rows.append({"field": col, "value": value, "count": int(count)})
    pd.DataFrame(rows).to_csv(path, index=False)


def evaluate_binary_models(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    raw_cols: list[str],
    structured_cols: list[str],
    control_cols: list[str],
    max_splits: int,
) -> pd.DataFrame:
    model_specs = [
        ("Majority baseline", None, []),
        ("LR (controls only)", logistic_regression(), control_cols),
        ("LR (raw gaze)", logistic_regression(), raw_cols),
        ("LR (structured gaze)", logistic_regression(), structured_cols),
        (
            "LR (structured gaze+controls)",
            logistic_regression(),
            structured_cols + control_cols,
        ),
    ]
    y = df[target_col].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(max_splits, pd.Series(groups).nunique())
    if n_splits < 2:
        raise ValueError(f"Need at least two groups for grouped CV over {group_col}")
    cv = GroupKFold(n_splits=n_splits)

    rows = []
    for name, model, cols in model_specs:
        if model is None:
            pred = np.full_like(y, majority_label(y))
        elif not cols:
            continue
        else:
            pred = cross_val_predict(model, df[cols].to_numpy(), y, cv=cv, groups=groups)
        rows.append(
            {
                "model": name,
                "rows": int(len(df)),
                "groups": int(pd.Series(groups).nunique()),
                "positive_rate": float(y.mean()),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "f1_positive": f1_score(y, pred, pos_label=1, zero_division=0),
                "f1_negative": f1_score(y, pred, pos_label=0, zero_division=0),
                "cv_splits": int(n_splits),
            }
        )
    return pd.DataFrame(rows)


def evaluate_feature_ablation(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    specs: list[tuple[str, list[str]]],
    max_splits: int,
) -> pd.DataFrame:
    y = df[target_col].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(max_splits, pd.Series(groups).nunique())
    cv = GroupKFold(n_splits=n_splits)

    rows = []
    for name, cols in specs:
        available = [col for col in cols if col in df.columns]
        X = df[available].fillna(0.0).to_numpy()
        pred = cross_val_predict(logistic_regression(), X, y, cv=cv, groups=groups)
        rows.append(
            {
                "feature_set": name,
                "n_features": len(available),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "f1_positive": f1_score(y, pred, pos_label=1, zero_division=0),
                "f1_negative": f1_score(y, pred, pos_label=0, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def lr_coefficients(df: pd.DataFrame, target_col: str, cols: list[str]) -> pd.DataFrame:
    """Standardized LR coefficients (log-odds per SD) fit on the full dataset."""
    available = [col for col in cols if col in df.columns]
    X = df[available].fillna(0.0).to_numpy()
    y = df[target_col].astype(int).to_numpy()
    model = logistic_regression().fit(X, y)
    coefs = model.named_steps["clf"].coef_[0]
    out = pd.DataFrame(
        {
            "feature": available,
            "coef_log_odds_per_sd": coefs,
            "odds_ratio_per_sd": np.exp(coefs),
        }
    )
    return out.reindex(out["coef_log_odds_per_sd"].abs().sort_values(ascending=False).index)


def logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def majority_label(y: np.ndarray) -> int:
    return int(y.mean() >= 0.5)


def mann_whitney_tests(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    positive = df[df[target_col] == 1]
    negative = df[df[target_col] == 0]
    rows = []
    for feature in feature_cols:
        u_stat, p_value = stats.mannwhitneyu(
            positive[feature],
            negative[feature],
            alternative="two-sided",
        )
        n_positive = len(positive)
        n_negative = len(negative)
        rank_biserial = (2 * u_stat / (n_positive * n_negative)) - 1
        rows.append(
            {
                "feature": feature,
                "positive_mean": positive[feature].mean(),
                "negative_mean": negative[feature].mean(),
                "u_stat": u_stat,
                "p_value": p_value,
                "rank_biserial_r": rank_biserial,
                "significant_p_lt_0_05": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def kruskal_tests(df: pd.DataFrame, label_col: str, feature_cols: list[str]) -> pd.DataFrame:
    labels = sorted(df[label_col].dropna().unique())
    rows = []
    for feature in feature_cols:
        groups = [df.loc[df[label_col] == label, feature].dropna() for label in labels]
        groups = [group for group in groups if len(group) > 0]
        if len(groups) < 2:
            continue
        h_stat, p_value = stats.kruskal(*groups)
        rows.append(
            {
                "feature": feature,
                "h_stat": h_stat,
                "p_value": p_value,
                "significant_p_lt_0_05": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def markdown_table(df: pd.DataFrame, float_cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, record in df.iterrows():
        values = []
        for header in headers:
            value = record[header]
            if header in float_cols and isinstance(value, (float, np.floating)):
                if math.isnan(float(value)):
                    values.append("nan")
                elif header in ("p_value", "q_value"):
                    values.append(f"{float(value):.3g}")
                else:
                    values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
