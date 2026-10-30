"""Data quality checks and cautious interpolation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def quality_report(df: pd.DataFrame, depth_col: str = "DEPTH") -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        missing = int(s.isna().sum())
        rows.append({
            "curve": col,
            "samples": len(s),
            "missing": missing,
            "missing_pct": round(100 * missing / max(len(s), 1), 2),
            "unique_values": int(s.nunique(dropna=True)),
            "constant_or_empty": bool(s.nunique(dropna=True) <= 1),
            "min": s.min() if s.notna().any() else np.nan,
            "max": s.max() if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def interpolate_short_gaps(df: pd.DataFrame, columns: list[str], limit: int = 3) -> pd.DataFrame:
    """Interpolate only short internal gaps; do not extrapolate at either end."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = values.interpolate(method="linear", limit=limit, limit_area="inside")
    return out
