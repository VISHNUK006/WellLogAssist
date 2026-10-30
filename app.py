from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.las_io import read_las
from src.qc import quality_report, interpolate_short_gaps
from src.interpretation import density_porosity, suggest_boundaries
from src.modeling import leave_one_group_out_validation, train_for_one_held_out_group

st.set_page_config(page_title="WellLogAssist", page_icon="🛢️", layout="wide")
st.title("🛢️ WellLogAssist")
st.caption("Well-log visualization, quality control and group-based lithology validation")
st.warning(
    "Educational prototype only. Predictions, porosity estimates and boundary suggestions "
    "require independent geological and petrophysical review."
)

with st.sidebar:
    st.header("Well-log input")
    uploaded = st.file_uploader("Upload a LAS file", type=["las"])
    use_demo = st.checkbox("Use synthetic demo LAS", value=uploaded is None)
    st.caption("The bundled LAS is synthetic and is only for demonstrating the interface.")

if uploaded is not None:
    df, metadata, curve_meta = read_las(uploaded.getvalue())
elif use_demo:
    with open("data/sample/demo_well_synthetic.las", "rb") as f:
        df, metadata, curve_meta = read_las(f.read())
else:
    st.info("Upload a LAS file or select the synthetic demo LAS.")
    st.stop()

df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")
df = df.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)
numeric_curves = [
    c for c in df.columns if c != "DEPTH"
    and pd.to_numeric(df[c], errors="coerce").notna().any()
]

st.subheader("Well overview")
a, b, c, d = st.columns(4)
a.metric("Samples", f"{len(df):,}")
b.metric("Log curves", len(numeric_curves))
c.metric("Top depth", f"{df.DEPTH.min():.2f}")
d.metric("Bottom depth", f"{df.DEPTH.max():.2f}")
with st.expander("Well metadata and curve dictionary"):
    st.json(metadata)
    st.dataframe(pd.DataFrame(curve_meta), use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Log viewer", "Quality control", "Exploratory interpretation", "Unseen-area ML validation"]
)

with tab1:
    selected = st.multiselect(
        "Curves to display", numeric_curves,
        default=numeric_curves[:min(4, len(numeric_curves))]
    )
    if selected:
        fig = make_subplots(rows=1, cols=len(selected), shared_yaxes=True,
                            horizontal_spacing=0.025, subplot_titles=selected)
        for i, curve in enumerate(selected, start=1):
            fig.add_trace(
                go.Scatter(x=pd.to_numeric(df[curve], errors="coerce"),
                           y=df.DEPTH, mode="lines", name=curve,
                           line={"width": 1.2}),
                row=1, col=i
            )
            fig.update_xaxes(title_text=curve, row=1, col=i)
        fig.update_yaxes(title_text="Measured depth", autorange="reversed")
        fig.update_layout(height=720, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Curve quality report")
    st.dataframe(quality_report(df), use_container_width=True)
    st.download_button(
        "Download QC report",
        quality_report(df).to_csv(index=False).encode("utf-8"),
        "well_log_qc_report.csv", "text/csv"
    )
    st.subheader("Short-gap interpolation preview")
    cols = st.multiselect("Curves for short-gap interpolation", numeric_curves, key="interp_cols")
    limit = st.number_input("Maximum internal gap length", 1, 20, 3)
    if st.button("Create interpolation preview"):
        preview = interpolate_short_gaps(df, cols, int(limit))
        st.dataframe(preview.head(50), use_container_width=True)
        st.download_button(
            "Download interpolated preview",
            preview.to_csv(index=False).encode("utf-8"),
            "well_log_interpolation_preview.csv", "text/csv"
        )

with tab3:
    st.subheader("Density porosity estimate")
    density_candidates = [c for c in numeric_curves if any(k in c for k in ("RHOB", "RHO", "DEN"))]
    choices = density_candidates or numeric_curves
    if choices:
        rhob = st.selectbox("Bulk-density curve", choices)
        c1, c2 = st.columns(2)
        matrix_density = c1.number_input("Matrix density (g/cc)", 1.0, 4.0, 2.65, 0.01)
        fluid_density = c2.number_input("Fluid density (g/cc)", 0.1, 2.0, 1.00, 0.01)
        phi = density_porosity(df[rhob], matrix_density, fluid_density)
        fig = go.Figure(go.Scatter(x=phi, y=df.DEPTH, mode="lines", name="Density porosity"))
        fig.update_yaxes(autorange="reversed", title="Measured depth")
        fig.update_xaxes(title="Density porosity (fraction)")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Simple density-porosity equation. Validate matrix/fluid assumptions, units, shale, gas and borehole effects.")

    st.subheader("Candidate formation-boundary suggestions")
    boundary_curves = st.multiselect("Curves for change detection", numeric_curves, default=numeric_curves[:2])
    sensitivity = st.slider("Change sensitivity", 1.0, 5.0, 2.5, 0.1)
    if st.button("Suggest candidate boundaries"):
        st.session_state["candidate_boundaries"] = suggest_boundaries(
            df, "DEPTH", boundary_curves, sensitivity=sensitivity
        )
    if "candidate_boundaries" in st.session_state:
        st.dataframe(st.session_state["candidate_boundaries"], use_container_width=True)
        st.caption("Heuristic candidates only; not confirmed formation tops.")

with tab4:
    st.header("Test generalization to an unseen geological area")
    st.markdown(
        "Upload a **labelled multi-area CSV**. Choose the area column as the group column. "
        "The application holds out one complete area at a time. Imputation and model fitting "
        "happen only on the training areas."
    )
    st.info(
        "For unseen-area claims, group by AREA_ID (not individual depth rows). "
        "For unseen-well claims within a known area, group by WELL_ID. "
        "Do not use AREA_ID, WELL_ID, DEPTH, sample identifiers or target-derived columns as features."
    )
    training_file = st.file_uploader("Upload labelled multi-area CSV", type=["csv"], key="area_csv")
    if training_file:
        train_df = pd.read_csv(training_file)
        st.write("Data preview")
        st.dataframe(train_df.head(10), use_container_width=True)
        columns = train_df.columns.tolist()
        target_default = columns.index("LITHOLOGY") if "LITHOLOGY" in columns else len(columns) - 1
        target_col = st.selectbox("Lithology target column", columns, index=target_default)
        group_candidates = [c for c in columns if c != target_col and train_df[c].nunique(dropna=True) > 1]
        suggested_group = "AREA_ID" if "AREA_ID" in columns else ("STUDY_AREA" if "STUDY_AREA" in columns else (group_candidates[0] if group_candidates else columns[0]))
        group_col = st.selectbox("Group column (area identifier for area transfer)", group_candidates,
                                 index=group_candidates.index(suggested_group) if suggested_group in group_candidates else 0)
        numeric_candidates = [
            c for c in columns if c not in {target_col, group_col}
            and pd.api.types.is_numeric_dtype(train_df[c])
        ]
        # Depth and identifier fields are excluded from defaults.
        safe_defaults = [
            c for c in numeric_candidates
            if c.upper() not in {"DEPTH", "MD", "TVD", "X", "Y", "LAT", "LON", "LONGITUDE", "LATITUDE"}
            and not any(token in c.upper() for token in ("_ID", "SAMPLE_ID", "WELL_ID", "AREA_ID", "GID"))
        ]
        features = st.multiselect("Numeric log features", numeric_candidates, default=safe_defaults)
        if any(c.upper() in {"DEPTH", "MD", "TVD"} for c in features):
            st.error("Remove depth features for this default cross-area experiment; depth can create area-specific shortcuts.")
        if group_col in features or target_col in features:
            st.error("The group and target columns must not be model features.")
        st.write("Number of groups:", int(train_df[group_col].nunique(dropna=True)))
        st.write("Lithology classes:", int(train_df[target_col].nunique(dropna=True)))
        st.warning(
            "A valid area-transfer test needs multiple genuinely independent areas, reliable labels, "
            "comparable log mnemonics/units and adequate class coverage. If a class exists only in the "
            "held-out area, a closed-set classifier cannot predict it; the app reports it as unseen."
        )
        if st.button("Run Leave-One-Area-Out validation", type="primary",
                     disabled=(not features or group_col in features or target_col in features or any(c.upper() in {"DEPTH", "MD", "TVD"} for c in features))):
            try:
                summary, predictions = leave_one_group_out_validation(
                    train_df, target_col, features, group_col
                )
                st.session_state["area_summary"] = summary
                st.session_state["area_predictions"] = predictions
            except Exception as exc:
                st.error(str(exc))

    if "area_summary" in st.session_state:
        summary = st.session_state["area_summary"]
        predictions = st.session_state["area_predictions"]
        st.subheader("Area-wise results")
        st.dataframe(summary, use_container_width=True)
        successful = summary[summary["status"] == "OK"]
        if len(successful):
            avg_macro = successful["macro_f1_known_classes"].mean()
            avg_acc = successful["accuracy_known_classes"].mean()
            x1, x2 = st.columns(2)
            x1.metric("Mean area macro F1 (known classes)", f"{avg_macro:.3f}")
            x2.metric("Mean area accuracy (known classes)", f"{avg_acc:.3f}")
            st.caption("Unweighted mean across successfully evaluated areas. Also inspect each area's result and sample count.")
        st.download_button(
            "Download area-wise metrics",
            summary.to_csv(index=False).encode("utf-8"),
            "leave_one_area_out_metrics.csv", "text/csv"
        )
        st.download_button(
            "Download out-of-fold predictions",
            predictions.to_csv(index=False).encode("utf-8"),
            "leave_one_area_out_predictions.csv", "text/csv"
        )
        st.caption(
            "Metrics are computed only on test samples whose lithology class occurs in that fold's training data. "
            "Unseen classes and samples are reported separately, not silently treated as known-class predictions."
        )
