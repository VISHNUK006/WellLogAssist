
"""LAS parsing helpers."""
from __future__ import annotations

from io import StringIO
import lasio
import pandas as pd


def read_las(source) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Read a LAS file path or uploaded bytes into a DataFrame."""

    if isinstance(source, (bytes, bytearray)):
        try:
            text = source.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = source.decode("latin-1")

        las = lasio.read(StringIO(text))

    elif hasattr(source, "read"):
        content = source.read()

        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                content = content.decode("latin-1")

        las = lasio.read(StringIO(content))

    else:
        las = lasio.read(source)

    df = las.df().reset_index()

    if df.empty:
        raise ValueError("The LAS file contains no curve samples.")

    depth_col = df.columns[0]
    df = df.rename(columns={depth_col: "DEPTH"})
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    metadata = {
        "well_name": str(getattr(las.well, "WELL", "") or ""),
        "uwi": str(getattr(las.well, "UWI", "") or ""),
        "company": str(getattr(las.well, "COMP", "") or ""),
        "field": str(getattr(las.well, "FLD", "") or ""),
        "null_value": getattr(las, "null", None),
        "start": las.well["STRT"].value if las.well["STRT"] else None,
        "stop": las.well["STOP"].value if las.well["STOP"] else None,
        "step": las.well["STEP"].value if las.well["STEP"] else None,
    }

    curve_metadata = []

    for curve in las.curves:
        curve_metadata.append({
            "mnemonic": str(curve.mnemonic).upper(),
            "unit": str(curve.unit or ""),
            "description": str(curve.descr or ""),
        })

    return df, metadata, curve_metadata
