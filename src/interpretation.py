"""Transparent, heuristic log interpretation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd


def density_porosity(rhob: pd.Series, matrix_density: float = 2.65,
                     fluid_density: float = 1.0) -> pd.Series:
    """Calculate simple density porosity fraction; inputs must use consistent units."""
    if matrix_density <= fluid_density:
        raise ValueError("Matrix density must be greater than fluid density.")
    rhob = pd.to_numeric(rhob, errors="coerce")
    phi = (matrix_density - rhob) / (matrix_density - fluid_density)
    return phi.clip(lower=0, upper=1)


def suggest_boundaries(df: pd.DataFrame, depth_col: str,
                       curves: list[str], window: int = 7,
                       min_distance: int = 5, sensitivity: float = 2.0) -> pd.DataFrame:
    """Suggest candidate boundaries where rolling curve changes are unusually large.

    This is a simple robust-change heuristic, not a stratigraphic interpretation.
    """
    if depth_col not in df or not curves:
        return pd.DataFrame(columns=["DEPTH", "CURVE", "CHANGE_SCORE"])
    results = []
    for curve in curves:
        if curve not in df:
            continue
        x = pd.to_numeric(df[curve], errors="coerce")
        smooth = x.rolling(window=max(3, window), center=True, min_periods=2).median()
        delta = smooth.diff().abs()
        med = delta.median(skipna=True)
        mad = (delta - med).abs().median(skipna=True)
        scale = 1.4826 * mad if pd.notna(mad) and mad > 0 else (delta.std(skipna=True) or 0)
        if not scale or not np.isfinite(scale):
            continue
        score = (delta - med) / scale
        candidates = np.where(score.fillna(0).to_numpy() >= sensitivity)[0]
        last = -10**9
        for idx in candidates:
            if idx >= len(df) or idx - last < min_distance:
                continue
            depth = pd.to_numeric(df[depth_col], errors="coerce").iloc[idx]
            if pd.notna(depth):
                results.append({"DEPTH": float(depth), "CURVE": curve,
                                "CHANGE_SCORE": float(score.iloc[idx])})
                last = idx
    return pd.DataFrame(results).sort_values("DEPTH").reset_index(drop=True) if results else pd.DataFrame(columns=["DEPTH", "CURVE", "CHANGE_SCORE"])
