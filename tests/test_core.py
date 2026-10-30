import pandas as pd
import pytest
from src.interpretation import density_porosity, suggest_boundaries
from src.qc import quality_report, interpolate_short_gaps


def test_density_porosity_basic():
    rhob = pd.Series([2.65, 2.325, 2.0])
    result = density_porosity(rhob, 2.65, 1.0)
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.2)
    assert result.iloc[2] == pytest.approx(0.4)


def test_density_porosity_invalid_densities():
    with pytest.raises(ValueError):
        density_porosity(pd.Series([2.3]), 1.0, 1.0)


def test_qc_missing_values():
    df = pd.DataFrame({"DEPTH": [1, 2, 3], "GR": [10, None, 30]})
    report = quality_report(df)
    gr = report.loc[report.curve == "GR"].iloc[0]
    assert gr["missing"] == 1


def test_interpolate_short_internal_gap():
    df = pd.DataFrame({"GR": [10.0, None, 30.0, None, None, None, 70.0]})
    out = interpolate_short_gaps(df, ["GR"], limit=1)
    assert out["GR"].iloc[1] == pytest.approx(20.0)
    assert pd.isna(out["GR"].iloc[3])


def test_boundary_function_empty_when_no_curves():
    df = pd.DataFrame({"DEPTH": [1, 2, 3], "GR": [10, 11, 12]})
    out = suggest_boundaries(df, "DEPTH", [])
    assert out.empty
