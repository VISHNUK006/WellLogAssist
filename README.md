# WellLogAssist

**An educational well-log visualization, quality-control and cross-area lithology validation application.**

WellLogAssist reads LAS well logs, visualizes curves, summarizes data quality, provides transparent exploratory calculations, and evaluates a supervised lithology classifier using **Leave-One-Group-Out (LOGO) validation**.

> **Scientific integrity:** This repository is a prototype. It does not establish geological or petrophysical truth. The bundled LAS is synthetic. The application does not include a real multi-area labelled well-log dataset, so it cannot provide evidence of generalization until you supply suitable independent data.

## Main capabilities

- LAS upload and curve/metadata inspection.
- Multi-track log visualization against measured depth.
- QC summary for missing values, unique values and ranges.
- Short internal-gap interpolation preview.
- Simple density-porosity calculation with explicit matrix/fluid assumptions.
- Heuristic candidate boundary suggestions.
- Optional Random Forest lithology classification from a labelled CSV.
- **Leave-One-Area-Out / Leave-One-Group-Out evaluation**:
  - one complete area is held out per fold;
  - all other areas are used for training;
  - imputation is fitted only on training data;
  - each area gets its own metrics;
  - classes absent from training are identified as unseen;
  - out-of-fold predictions and metrics can be downloaded.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Use the application

1. Upload a LAS file, or select the synthetic demo LAS.
2. Inspect the log viewer and QC report.
3. For ML validation, upload a separate labelled multi-area CSV.
4. Select the lithology target column.
5. Select the **area identifier** as the group column (for example `AREA_ID` or `STUDY_AREA`).
6. Select numeric log measurements as features.
7. Run Leave-One-Area-Out validation.
8. Download area-wise metrics and out-of-fold predictions.

## Labelled CSV requirements

One row per depth sample, with at least:

- an area/group identifier (for example `AREA_ID`);
- a categorical target (for example `LITHOLOGY`);
- numeric log curves (for example `GR`, `RHOB`, `NPHI`, `RT`).

Example schema only:

```csv
AREA_ID,WELL_ID,DEPTH,GR,RHOB,NPHI,RT,LITHOLOGY
Area_A,Well_A1,1000,72.4,2.51,0.19,8.2,Sandstone
Area_A,Well_A1,1000.5,68.1,2.48,0.21,7.5,Sandstone
Area_B,Well_B1,1500,112.7,2.66,0.12,2.1,Shale
Area_B,Well_B1,1500.5,118.3,2.68,0.10,1.8,Shale
```

The rows above are illustrative and are not a real dataset. Ensure that curve mnemonics, units, depth references and lithology labels are consistent across wells/areas. Record data sources and licensing.

## Validation design: what does "unseen area" mean?

If the scientific question is transfer to a **new geological area**, use `AREA_ID` (or equivalent) as the group column. The software uses `LeaveOneGroupOut`: in each fold, every sample from one area is the test set and no sample from that area is used for model fitting.

If the question is transfer to a **new well within an already represented area**, group by `WELL_ID` instead. This is a different and generally easier generalization question.

### Leakage controls

- Area/group ID and target are prohibited as model features.
- The app's default feature selection excludes depth and common identifier columns.
- Median imputation is inside the sklearn Pipeline, so it is fitted only on each training fold.
- There is no random row-level split in the area-validation workflow.
- Out-of-fold predictions are generated only for held-out groups.

Before reporting results, also inspect the input data for duplicate samples, duplicated wells across area labels, repeated or derived target information, inconsistent curve units, and any preprocessing performed upstream using all areas.

### Metrics and unseen classes

For each held-out group the app reports:

- `accuracy_known_classes`
- `macro_f1_known_classes`
- `weighted_f1_known_classes`
- number of test samples whose class was present in training
- number of test samples whose class was absent from training
- names of unseen classes

Known-class metrics are calculated only on test samples whose lithology occurs in that fold's training data. This avoids pretending a closed-set classifier can predict a class it has never seen. **Always report the unseen-class count alongside the metrics.** A high known-class score with many unseen classes is not evidence of broad geological transfer.

The application also shows the unweighted mean across successfully evaluated areas. Do not report only this average: provide each area's score, test sample count, class distribution and unseen-class count.

## Scientific limitations

- A closed-set Random Forest cannot discover a lithology class absent from its training labels. Such classes are reported as unseen.
- LOGO validation estimates transfer only across the supplied groups. Three areas provide only three held-out folds; results can be highly sensitive to area composition.
- If there is only one area, unseen-area generalization cannot be evaluated.
- If lithology labels are sparse, inconsistent or derived from the same logs being used as predictors, evaluation may be misleading.
- Different logging tools, borehole conditions, acquisition vintages, units and processing workflows can create domain shifts.
- Random Forest is a baseline, not a claim of state-of-the-art performance.
- Density porosity is a simple equation and does not correct for shale, gas, invasion, borehole effects or complex mineralogy.
- Candidate boundaries are change-point suggestions, not confirmed formation tops.

## Tests

```bash
pip install pytest
pytest -q
```

Tests include group isolation, one-fold-per-area behavior, unseen-class reporting and validation input checks.

## Repository layout

```text
WellLogAssist/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── las_io.py
│   ├── qc.py
│   ├── interpretation.py
│   └── modeling.py
├── tests/
│   ├── test_core.py
│   └── test_modeling.py
└── data/
    └── sample/
        ├── demo_well_synthetic.las
        └── training_template.csv
```

## Data statement

`data/sample/demo_well_synthetic.las` is generated synthetic data for software demonstration only. It is not a real well, does not represent a real basin or field, and must not be used to claim geological performance. No real labelled multi-area dataset is bundled.

## Disclaimer

Provided as-is for education and portfolio development. It is not a substitute for professional geological or petrophysical interpretation.
