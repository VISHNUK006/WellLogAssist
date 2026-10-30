"""Leakage-aware lithology modelling with group/area holdout validation.

For claims about transfer to a new geological area, set group_col to the
AREA identifier. Every sample from the held-out area is excluded from model
fitting, including imputation. For transfer to a new well in a known area,
group by WELL_ID instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut


def _validate_inputs(data: pd.DataFrame, target_col: str,
                     feature_cols: list[str], group_col: str):
    required = [target_col, group_col, *feature_cols]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if target_col == group_col:
        raise ValueError("Target and group columns must be different.")
    if target_col in feature_cols or group_col in feature_cols:
        raise ValueError("Target and group identifiers cannot be model features.")
    if not feature_cols:
        raise ValueError("Select at least one numeric log feature.")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("Feature list contains duplicates.")


def _make_pipeline(random_state: int = 42) -> Pipeline:
    # Imputation is fitted inside each training fold, never on held-out data.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def leave_one_group_out_validation(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    group_col: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each group once as a completely held-out test group.

    Returns:
      summary: one row per held-out group, with metrics and unseen-class count.
      predictions: out-of-fold predictions, one per evaluated sample.

    Test samples with a class never seen in that fold's training groups are
    explicitly marked as unseen and are not silently counted as model errors
    for the known-class-only metrics. Their count and share are reported.
    """
    _validate_inputs(data, target_col, feature_cols, group_col)

    work = data[[group_col, target_col, *feature_cols]].copy()
    work = work.dropna(subset=[group_col, target_col])
    work[group_col] = work[group_col].astype(str)
    work[target_col] = work[target_col].astype(str).str.strip()
    work = work[work[target_col] != ""].reset_index().rename(columns={"index": "SOURCE_INDEX"})

    groups = work[group_col].unique()
    if len(groups) < 2:
        raise ValueError(
            f"Need at least 2 distinct values in '{group_col}' for grouped validation; found {len(groups)}."
        )

    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    logo = LeaveOneGroupOut()
    summaries: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    X = work[feature_cols]
    y = work[target_col]
    g = work[group_col]

    for train_idx, test_idx in logo.split(X, y, groups=g):
        train = work.iloc[train_idx]
        test = work.iloc[test_idx]
        held_group = str(test[group_col].iloc[0])
        y_train = train[target_col].astype(str)
        y_test = test[target_col].astype(str)
        train_classes = sorted(y_train.unique())
        test_classes = sorted(y_test.unique())
        unseen_classes = sorted(set(test_classes) - set(train_classes))

        if len(train_classes) < 2:
            summaries.append({
                "held_out_group": held_group,
                "status": "SKIPPED: fewer than 2 training classes",
                "train_groups": int(train[group_col].nunique()),
                "train_samples": int(len(train)),
                "test_samples": int(len(test)),
                "known_test_samples": 0,
                "unseen_test_samples": int(len(test)),
                "unseen_test_classes": ", ".join(unseen_classes),
                "accuracy_known_classes": np.nan,
                "macro_f1_known_classes": np.nan,
                "weighted_f1_known_classes": np.nan,
            })
            continue

        model = _make_pipeline(random_state)
        model.fit(train[feature_cols], y_train)
        predicted = model.predict(test[feature_cols])

        known_mask = y_test.isin(train_classes).to_numpy()
        y_known = y_test.to_numpy()[known_mask]
        pred_known = predicted[known_mask]
        known_classes_in_test = sorted(set(y_known))

        if len(y_known):
            acc = accuracy_score(y_known, pred_known)
            macro = f1_score(
                y_known, pred_known, labels=known_classes_in_test,
                average="macro", zero_division=0
            )
            weighted = f1_score(
                y_known, pred_known, labels=known_classes_in_test,
                average="weighted", zero_division=0
            )
        else:
            acc = macro = weighted = np.nan

        summaries.append({
            "held_out_group": held_group,
            "status": "OK",
            "train_groups": int(train[group_col].nunique()),
            "train_samples": int(len(train)),
            "test_samples": int(len(test)),
            "known_test_samples": int(known_mask.sum()),
            "unseen_test_samples": int((~known_mask).sum()),
            "unseen_test_classes": ", ".join(unseen_classes),
            "accuracy_known_classes": acc,
            "macro_f1_known_classes": macro,
            "weighted_f1_known_classes": weighted,
        })

        pred_frame = test[["SOURCE_INDEX", group_col, target_col]].copy()
        pred_frame = pred_frame.rename(columns={
            group_col: "HELD_OUT_GROUP",
            target_col: "ACTUAL_LITHOLOGY",
        })
        pred_frame["PREDICTED_LITHOLOGY"] = predicted
        pred_frame["CLASS_SEEN_IN_TRAINING"] = known_mask
        pred_frame["FOLD_TRAINING_CLASSES"] = ", ".join(train_classes)
        prediction_frames.append(pred_frame)

    summary = pd.DataFrame(summaries)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else
        pd.DataFrame(columns=[
            "SOURCE_INDEX", "HELD_OUT_GROUP", "ACTUAL_LITHOLOGY",
            "PREDICTED_LITHOLOGY", "CLASS_SEEN_IN_TRAINING",
            "FOLD_TRAINING_CLASSES"
        ])
    )
    return summary, predictions


def train_for_one_held_out_group(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    group_col: str,
    held_out_group: str,
    random_state: int = 42,
):
    """Fit on all groups except one and return model plus transparent metrics."""
    _validate_inputs(data, target_col, feature_cols, group_col)
    work = data[[group_col, target_col, *feature_cols]].copy()
    work = work.dropna(subset=[group_col, target_col])
    work[group_col] = work[group_col].astype(str)
    work[target_col] = work[target_col].astype(str).str.strip()
    work = work[work[target_col] != ""]
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    held_out_group = str(held_out_group)
    train = work[work[group_col] != held_out_group]
    test = work[work[group_col] == held_out_group]
    if train.empty or test.empty:
        raise ValueError("Selected held-out group must exist and leave at least one training group.")
    if train[target_col].nunique() < 2:
        raise ValueError("Training groups must contain at least two lithology classes.")

    model = _make_pipeline(random_state)
    model.fit(train[feature_cols], train[target_col])
    pred = model.predict(test[feature_cols])
    known_classes = set(train[target_col].unique())
    known_mask = test[target_col].isin(known_classes).to_numpy()
    unseen_classes = sorted(set(test[target_col].unique()) - known_classes)
    y_known = test[target_col].to_numpy()[known_mask]
    p_known = pred[known_mask]
    labels = sorted(set(y_known))

    metrics = {
        "held_out_group": held_out_group,
        "train_groups": int(train[group_col].nunique()),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "known_test_samples": int(known_mask.sum()),
        "unseen_test_samples": int((~known_mask).sum()),
        "unseen_test_classes": unseen_classes,
        "accuracy_known_classes": accuracy_score(y_known, p_known) if len(y_known) else np.nan,
        "macro_f1_known_classes": f1_score(y_known, p_known, labels=labels, average="macro", zero_division=0) if len(y_known) else np.nan,
        "weighted_f1_known_classes": f1_score(y_known, p_known, labels=labels, average="weighted", zero_division=0) if len(y_known) else np.nan,
        "classification_report": classification_report(y_known, p_known, labels=labels, zero_division=0, output_dict=True) if len(y_known) else {},
        "confusion_matrix": confusion_matrix(y_known, p_known, labels=labels) if len(y_known) else np.empty((0, 0)),
        "classes": labels,
    }
    pred_frame = test[[group_col, target_col]].copy().rename(columns={
        group_col: "HELD_OUT_GROUP", target_col: "ACTUAL_LITHOLOGY"
    })
    pred_frame["PREDICTED_LITHOLOGY"] = pred
    pred_frame["CLASS_SEEN_IN_TRAINING"] = known_mask
    return model, metrics, pred_frame
