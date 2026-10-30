import pandas as pd
import pytest

from src.modeling import leave_one_group_out_validation, train_for_one_held_out_group


def make_data():
    rows = []
    # Each area has the same two classes with clearly separated feature values.
    for area, offset in [("A", 0.0), ("B", 0.2), ("C", -0.1)]:
        for lith, base in [("Sandstone", 1.0), ("Shale", 10.0)]:
            for i in range(12):
                rows.append({
                    "AREA_ID": area,
                    "WELL_ID": f"{area}_W1",
                    "DEPTH": 1000 + i * 0.5,
                    "GR": base + offset + i * 0.001,
                    "RHOB": (2.4 if lith == "Sandstone" else 2.65) + offset * 0.01,
                    "LITHOLOGY": lith,
                })
    return pd.DataFrame(rows)


def test_logo_holds_out_each_area_once():
    df = make_data()
    summary, predictions = leave_one_group_out_validation(
        df, "LITHOLOGY", ["GR", "RHOB"], "AREA_ID"
    )
    assert set(summary["held_out_group"]) == {"A", "B", "C"}
    assert len(predictions) == len(df)
    assert predictions.groupby("HELD_OUT_GROUP").size().to_dict() == {
        "A": 24, "B": 24, "C": 24
    }
    assert (summary["status"] == "OK").all()


def test_group_and_target_cannot_be_features():
    df = make_data()
    with pytest.raises(ValueError):
        leave_one_group_out_validation(
            df, "LITHOLOGY", ["GR", "AREA_ID"], "AREA_ID"
        )


def test_requires_multiple_groups():
    df = make_data()
    df["AREA_ID"] = "ONLY_ONE"
    with pytest.raises(ValueError, match="at least 2"):
        leave_one_group_out_validation(
            df, "LITHOLOGY", ["GR", "RHOB"], "AREA_ID"
        )


def test_unseen_lithology_is_reported():
    df = make_data()
    # Put a lithology class only in area C.
    df.loc[df["AREA_ID"] == "C", "LITHOLOGY"] = "Dolomite"
    summary, predictions = leave_one_group_out_validation(
        df, "LITHOLOGY", ["GR", "RHOB"], "AREA_ID"
    )
    row = summary.loc[summary["held_out_group"] == "C"].iloc[0]
    assert row["unseen_test_samples"] == 24
    assert "Dolomite" in row["unseen_test_classes"]
    assert not predictions.loc[
        predictions["HELD_OUT_GROUP"] == "C", "CLASS_SEEN_IN_TRAINING"
    ].any()


def test_single_heldout_area_model_excludes_test_area():
    df = make_data()
    model, metrics, preds = train_for_one_held_out_group(
        df, "LITHOLOGY", ["GR", "RHOB"], "AREA_ID", "C"
    )
    assert metrics["held_out_group"] == "C"
    assert metrics["train_groups"] == 2
    assert set(preds["HELD_OUT_GROUP"]) == {"C"}
