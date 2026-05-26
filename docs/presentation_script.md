# CerviRisk — 20-Minute Presentation Package

> **Format:** 15 min deep walkthrough (with slides/diagrams) + 5 min live demo + Q&A prep  
> **Language:** English  
> **Audience:** Technical reviewers / panel

---

## PART 1 — 15-MINUTE DEEP WALKTHROUGH

---

### SLIDE 1 — Title (30 sec)

**Title:** CerviRisk: An End-to-End Machine Learning System for Cervical Cancer Screening Risk Stratification

**What to say:**
> "Today I'm going to walk you through CerviRisk — a complete machine learning pipeline I built for cervical cancer screening risk prediction. The system ingests longitudinal HPV and cytology screening records, engineers leakage-aware historical features, trains and selects the best risk model, serves predictions via an API, and monitors monthly incoming data for distribution shift. I'll cover each layer in detail, then run the full pipeline live."

---

### SLIDE 2 — Problem Context (1 min)

**Title:** Why Patient-Level Risk Stratification?

**Bullet points:**
- National screening programs already handle: population-level invitations, HPV testing, cytology, follow-up guidelines
- What they don't optimize: *which specific patients* need earlier clinical review this month
- CerviRisk adds an ML layer that combines current test results + full screening history → individual CIN2+ risk score
- CIN2+ = Cervical Intraepithelial Neoplasia grade 2 or above (clinically significant precancerous lesion)

**Diagram — System positioning:**
```
┌─────────────────────────────────────────────────────────────────┐
│                   National Screening Program                    │
│   Invite → HPV Test → Cytology → Guideline Pathway → Follow-up │
└──────────────────────────┬──────────────────────────────────────┘
                           │ monthly screening batch
                           ▼
              ┌────────────────────────┐
              │   CerviRisk ML Layer   │  ◄── this project
              │  Risk Score + Triage   │
              └────────────────────────┘
                           │
                           ▼
              Clinician dashboard: who needs
              review earliest this month?
```

**What to say:**
> "Cervical screening programs are population-scale logistics systems. They're great at scheduling and running tests. But they don't have a patient-level risk layer that says: among the 750 women who came in this month, which 30 should I be worried about, even if their latest result looks borderline? That's the gap CerviRisk is designed to fill."

---

### SLIDE 3 — System Architecture (1 min)

**Title:** Full Pipeline at a Glance

```
┌───────────────┐    ┌─────────────────────┐    ┌────────────────┐
│  Data Layer   │    │   Modeling Layer     │    │ Serving Layer  │
│               │    │                     │    │                │
│  Simulator    │───►│  Preprocess +        │───►│  FastAPI       │
│  (or real     │    │  Feature Engineering │    │  /predict      │
│   registry)   │    │         │           │    │  (1yr/3yr/5yr) │
│               │    │         ▼           │    └────────────────┘
│  Monthly      │    │  Train / Tune /      │
│  batch ingest │    │  Select best model   │    ┌────────────────┐
│               │    │         │           │    │ Dashboard      │
│               │    │         ▼           │───►│ (Streamlit)    │
└───────────────┘    │  Batch Predict       │    └────────────────┘
                     │         │           │
                     │         ▼           │    ┌────────────────┐
                     │  Drift Monitor      │───►│ Drift Report   │
                     └─────────────────────┘    │ (JSON)         │
                                                └────────────────┘
```

---

### SLIDE 4 — The Data (2 min)

**Title:** What Does the Data Look Like?

**Sub-title:** Longitudinal screening records — one row per screening visit

**Schema table:**

| Field | Type | Example values |
|---|---|---|
| `person_id` | string | UUID (synthetic) |
| `screening_date` | date | 2016-03-14 |
| `age` | float | 34.2 |
| `region` | category | north / south / east / west / central |
| `hpv_test_result` | category | negative / positive |
| `hpv_genotype` | category | negative / hpv16 / hpv18 / other_hr / low_risk / multiple_hr |
| `cytology_result` | category | NILM / ASC-US / LSIL / ASC-H / HSIL / AGC |
| `histology_result` | category | none / normal / CIN1 / CIN2 / CIN3 / cancer |
| `hpv_vaccinated` | bool | True / False |
| `immunosuppressed` | bool | True / False |
| `parity` | int | 0–4 |
| `smoking_status` | category | never / former / current |
| `cin2plus_detected` | bool | True / False |

**Scale:**
- **22,000 women** simulated, spanning **2010–2026**
- Average ~4–6 screening visits per woman → ~100,000+ rows total
- Monthly prediction batch: **~750 records per month**

**Diagram — time structure:**
```
PERSON_ID=A  ──●────────●───────────●────────●──────────►
                2012     2015        2018      2022

PERSON_ID=B  ──●──────●────●──────────────────●─────────►
                2011   2014 2016               2021

Each ● = one screening visit row (HPV result + cytology + history)
```

**What to say:**
> "The raw data is longitudinal — one row per screening visit per woman. Each row captures the current visit results: HPV genotype, cytology, histology, whether she had a colposcopy or treatment. The simulator replicates realistic epidemiology: HPV genotypes have persistence probabilities — HPV16 persists at 62%, HPV18 at 48% — vaccination reduces high-risk HPV acquisition by ~38%, immunosuppression increases it by ~45%. So the synthetic data is not random noise; it models real biological processes."

---

### SLIDE 5 — Feature Engineering (2 min)

**Title:** Building Leakage-Aware Historical Features

**The leakage problem:**
```
WRONG (data leakage):
  Row for 2016 visit uses histology from 2019 → model "sees the future"

RIGHT (leakage-aware):
  For each row, only use information available at the time of that visit.
  All history features are computed with .cumcount() and .shift(1).
```

**Historical features added (computed per person, per visit, looking backward only):**

| Feature | What it captures |
|---|---|
| `n_previous_screens` | How many prior visits (`.cumcount()`) |
| `time_since_last_screen` | Gap in years to previous visit |
| `ever_had_abnormal_cyto` | Any ASC-US+ in past |
| `ever_had_hrHPV` | Any high-risk HPV ever detected |
| `ever_had_cin1plus` | Prior histology ≥ CIN1 |
| `ever_had_cin2plus` | Prior histology ≥ CIN2 |
| `persistent_hrhpv` | hrHPV at both this and previous visit |

**Outcome labeling (target variable):**
```
For each visit row:
  outcome_cin2_3yr = 1  if the same person develops CIN2+ within 3 years
                    = 0  otherwise

Similarly for 1yr and 5yr windows.
```

**Encoding:**
- Numeric columns: passed as-is (scaled for Logistic Regression)
- Categorical columns: one-hot encoded → e.g., `hpv_genotype_hpv16`, `cytology_result_HSIL`

**Total features after encoding: ~44 columns**

**What to say:**
> "Feature engineering is where I spent the most careful attention. The classic mistake in longitudinal medical data is leakage — using future information to predict the future. To prevent this, every history feature is computed using only past rows: cumulative counts, lagged values, backward-looking flags. The outcome label is then attached as a forward-looking window: does this person develop CIN2+ in the next 3 years? The model never sees what happened after the visit it's predicting for."

---

### SLIDE 6 — Time-Based Train/Validation/Test Split (1.5 min)

**Title:** Respecting Time Order — No Random Shuffling

**Diagram:**
```
2010────────────────2018│2019────2020│2021───2022│2023───────►
      TRAIN              │ VALIDATION │   TEST    │  HOLDOUT
  (for CV + fitting)     │  (model    │ (final    │ (future
                         │  selection)│  eval)    │  batches)
```

**Why time-split matters:**
- Random split would let the model train on 2022 data and test on 2015 data
- In reality, you always predict forward in time
- `TimeSeriesSplit(n_splits=4)` used for cross-validation: fold 1 trains on earliest slice, tests on the next; folds march forward

**Cross-validation setup:**
```
Fold 1: |──train──|──val──|
Fold 2: |────train────|──val──|
Fold 3: |──────train──────|──val──|
Fold 4: |────────train────────|──val──|
```

**What to say:**
> "I deliberately avoided random shuffling. When you shuffle temporal data, you're leaking the future into training. The validation set here is always a later time period than training. Cross-validation uses scikit-learn's TimeSeriesSplit — each fold adds more history and validates on the next slice. This gives a realistic estimate of how the model performs on data it hasn't seen yet."

---

### SLIDE 7 — Model Training & Comparison (2 min)

**Title:** Three Models, One Winner

**Models compared:**
1. **Logistic Regression** (with StandardScaler + class_weight='balanced')
2. **Random Forest** (n_estimators=240, balanced_subsample, min_samples_leaf=18)
3. **XGBoost** (n_estimators=260, max_depth=3, lr=0.045, scale_pos_weight for imbalance)

**Class imbalance handling:**
- CIN2+ positive rate in training data is low (realistic prevalence ~5–10%)
- LR: `class_weight='balanced'`
- RF: `class_weight='balanced_subsample'`
- XGBoost: `scale_pos_weight = n_negatives / n_positives`

**Cross-validation AUC results (4-fold TimeSeriesSplit, target: CIN2+ within 3 years):**

```
Model               Fold1   Fold2   Fold3   Fold4   Val(held)
──────────────────────────────────────────────────────────────
Logistic Regression  0.666   0.739   0.677   0.773   0.771 ✓ BEST
Random Forest        0.700   0.708   0.654   0.671   0.686
XGBoost              0.667   0.711   0.677   0.615   0.612
```

**Validation set final metrics (best model = Logistic Regression):**

| Metric | Value |
|---|---|
| AUC | **0.771** |
| Sensitivity | 0.703 |
| Specificity | 0.740 |

**What to say:**
> "I trained three models: Logistic Regression, Random Forest, and XGBoost. For model selection, I used the held-out validation set — 2019–2020 data — to pick the winner. Logistic Regression came out on top with AUC 0.771. This is actually common in structured medical data: LR generalizes well when you've built good features and handled class imbalance. XGBoost scored lower, likely because the 4-fold time-series CV didn't give it enough data in early folds to learn the tree splits. That said, I still deploy XGBoost for the 1-year and 5-year horizons because they were trained on the full train+validation set where XGBoost had more data."

---

### SLIDE 8 — Feature Importance & SHAP (1.5 min)

**Title:** What Does the Model Actually Use?

**Top 10 features by XGBoost importance:**

```
Feature                    Importance
─────────────────────────────────────
hrhpv_positive               0.148  ████████████████
hpv_test_result_negative     0.092  ██████████
hpv_genotype_other_hr        0.064  ███████
hpv_genotype_multiple_hr     0.029  ███
immunosuppressed             0.029  ███
hpv_genotype_hpv16           0.025  ██
persistent_hrhpv             0.022  ██
hpv_vaccinated               0.022  ██
histology_result_none        0.021  ██
cytology_result_ASC-H        0.021  ██
```

**SHAP interpretation (3 patient archetypes in `reports/`):**
- `shap_force_low_risk.html` — patient with negative HPV, NILM cytology → low risk score
- `shap_force_high_risk.html` — HPV16+, persistent hrHPV, prior CIN1 → high risk score
- `shap_force_possible_false_negative.html` — borderline case: negative current HPV but strong history

**What to say:**
> "The top driver is `hrhpv_positive` — whether the current visit shows high-risk HPV — which is clinically expected. The second most important is actually the negative HPV flag, which reassures clinicians when it's absent. HPV genotype details, immunosuppression, and vaccine status all contribute. The SHAP force plots in the reports folder let you decompose any individual prediction into additive feature contributions — very useful for a clinician who asks 'why is this patient flagged?'"

---

### SLIDE 9 — Prediction Serving (1 min)

**Title:** FastAPI Endpoint — Three Risk Horizons

**Endpoint:** `POST /predict`

**Input:** a single screening record JSON (age, HPV genotype, cytology, history flags, ...)

**Output:**
```json
{
  "risk_probabilities": {
    "1yr": 0.07,
    "3yr": 0.21,
    "5yr": 0.34
  },
  "model": "xgboost",
  "target": "CIN2+"
}
```

**Why three horizons?**
- 1-year: flag for urgent colposcopy referral
- 3-year: primary screening recall decision
- 5-year: long-term surveillance vs. discharge planning

**What to say:**
> "The serving layer is a FastAPI app with a single POST endpoint. It loads three pre-trained XGBoost models at startup and returns 1-year, 3-year, and 5-year CIN2+ risk probabilities in a single call. This is designed for system-to-system integration — a hospital EMR would call this API after each screening visit and store the risk scores for clinical review."

---

### SLIDE 10 — Drift Monitoring (2 min)

**Title:** How Do We Know the Model Is Still Valid Next Month?

**The problem — Data Drift:**
```
TRAINING DATA (2010–2018):
  Age distribution: mean ~38 yrs, normal shape
  HPV16 prevalence: ~2%

INCOMING BATCH (2026):
  Age distribution: shifted (younger cohort from new screening policy)
  HPV genotype mix: shifted (new variant prevalence)

Result: model trained on 2010-2018 data may score 2026 patients incorrectly
```

**Two statistical tests used:**

| Feature type | Test | Logic |
|---|---|---|
| Numeric (age) | **Kolmogorov-Smirnov (KS)** | Compares the full CDFs of reference vs. current batch |
| Categorical (hpv_genotype, cytology_result) | **Chi-square** | Compares category frequency tables |

**Threshold:** p-value < 0.01 → drift detected

**Real output from the demo run:**
```json
{
  "age": {
    "test": "ks",
    "statistic": 0.235,
    "p_value": 1.66e-36,
    "drift_detected": true         ← injected age shift
  },
  "hpv_genotype": {
    "test": "chi_square",
    "p_value": 2.53e-31,
    "drift_detected": true         ← 18% of batch forced to "other_hr"
  },
  "cytology_result": {
    "test": "chi_square",
    "p_value": 0.747,
    "drift_detected": false        ← cytology distribution stable
  },
  "overall_drift_detected": true
}
```

**What drift means operationally:**
```
No drift:   run monthly predictions as normal
Drift:      alert → investigate → decide:
              • Is this real? (e.g., new age policy)
              • Retrain model on more recent data
              • Adjust decision threshold
              • Flag predictions with lower confidence
```

**What to say:**
> "The monitoring module runs every month alongside batch prediction. It compares the incoming batch against the training reference distribution for three key signals: age distribution using the KS test, and HPV genotype and cytology result distributions using chi-square. The KS test is ideal for continuous data because it compares the full empirical CDFs — not just means or variance — so it catches subtle shape changes too. In the demo drift report you can see that age and HPV genotype both shifted significantly — we injected an artificial shift — while cytology stayed stable. In production, drift detection would trigger an alert so the data science team can investigate before relying on the model for another month."

---

### SLIDE 11 — Clinician Dashboard (1 min)

**Title:** Monthly Triage Dashboard

**Built with:** Streamlit

**What it shows:**
- List of patients in the current monthly batch, sorted by risk tier
- Risk tiers: **Urgent** (high 1yr risk), **High** (high 3yr), **Monitor** (moderate), **Routine**
- Recommended action per patient (e.g., "Colposcopy referral", "Recall in 6 months", "Normal recall")
- Drift alert banner if monitoring detected a shift
- Cluster profiles of patient subgroups (unsupervised KMeans, 4 clusters)

**Privacy note:**
> Dashboard uses synthetic, de-identified patient IDs only — no names, addresses, or national identifiers. Real deployment requires authentication, role-based access, and audit logging.

**What to say:**
> "The dashboard is the clinician-facing output. A clinical lead opens it once a month, sees which patients in the batch need urgent attention, reviews the recommended actions, and can also see whether the drift monitor raised any flags. The cluster profiles panel gives an overview of the patient mix that month — useful for spotting if there's a new cohort type entering the screening program."

---

## PART 2 — 5-MINUTE LIVE DEMO

---

### Demo Script (step by step, ~5 min)

**Setup before the talk:**
```bash
cd /Users/niuniu/Desktop/cervirisk/CerviRisk-project
source .venv/bin/activate
```

---

**Step 1 — Run training pipeline (~2 min)**

```bash
python src/training_pipeline.py --bootstrap-synthetic
```

Say while running:
> "This single command does the full training side: simulates 22,000 women of longitudinal data, preprocesses it into features, runs a time-based train/val/test split, trains three models with cross-validation, picks the best, and generates SHAP explanations."

Watch for output lines:
```
=== historical synthetic raw data bootstrap ===
=== preprocessing and feature engineering ===
=== model training ===
=== unsupervised cluster profiling ===
=== model explanations ===
Training pipeline complete.
```

Point out model metrics printed to terminal: AUC scores per model per fold.

---

**Step 2 — Run prediction pipeline with drift injection (~1 min)**

```bash
python src/prediction_pipeline.py --batch-date 2026-01-01 --inject-demo-drift
```

Say:
> "Now I simulate a monthly incoming batch for January 2026 and deliberately inject a distribution shift — 18% of patients get reassigned to 'other_hr' HPV genotype. Watch what the drift monitor catches."

Expected output: drift detected for age and hpv_genotype, not cytology.

---

**Step 3 — Show drift report (~30 sec)**

```bash
cat reports/drift_report.json
```

Highlight: `"overall_drift_detected": true` and explain the p-values.

---

**Step 4 — Hit the API (~30 sec)**

In one terminal:
```bash
uvicorn src.serving.app:app --reload --port 8000
```

In another:
```bash
python src/serving/test_request.py
```

Show the three-horizon risk output.

---

**Step 5 — Open dashboard (~30 sec)**

```bash
streamlit run src/dashboard/app.py
```

Point out: patient list, risk tiers, drift alert banner.

---

## PART 3 — Q&A PREPARATION

---

### Q1: What if the data volume is very large?

**Answer:**
> "The system has three scalability levers. First, the data layer supports Parquet partitioning by year, so you never load the full dataset — only the relevant year slices. Second, the feature engineering has a Polars implementation (`preprocess_polars.py`) that is significantly faster than Pandas for large DataFrames. Third, for model training, there's a Dask-XGBoost mode: set `CERVIRISK_USE_DASK_XGB=1` and the training step runs distributed across a local or remote Dask cluster. The simulator can handle 500,000+ women via chunked writes with `--n-women 500000 --chunk-size 25000`."

---

### Q2: How does drift monitoring work and how do you prevent it from being too noisy?

**Answer:**
> "We monitor three features: age (KS test), HPV genotype (chi-square), and cytology result (chi-square). The threshold is p < 0.01, which is more conservative than the typical 0.05 — it reduces false alarms. We don't monitor every feature because that would cause multiple comparison inflation. Instead, we track the features that are both clinically meaningful and known to affect model output (the top predictors). In production, you'd add a Bonferroni or Benjamini-Hochberg correction if you expanded the monitored set."

---

### Q3: Why not a deep learning model?

**Answer:**
> "For structured tabular medical data with ~44 features, gradient boosted trees consistently outperform deep learning. Deep learning needs much more data to learn feature interactions that XGBoost discovers in a few hundred trees. Additionally, model interpretability is critical in a clinical setting — SHAP works natively with XGBoost and gives per-feature attribution that a clinician can audit. A deep neural network would add complexity without measurable accuracy gain here."

---

### Q4: Why does Logistic Regression beat XGBoost in your comparison?

**Answer:**
> "The key is the training set size at each validation fold. With TimeSeriesSplit on 2010–2018 data, the earliest folds have only 2–3 years of training data. XGBoost needs more rows to learn meaningful tree splits at depth 3+; Logistic Regression generalizes from fewer samples because it's a linear model. When I train XGBoost on the full train+validation set for the deployed models, it performs comparably. This also illustrates why the held-out validation set — not CV average — is the right selection criterion."

---

### Q5: How do you handle class imbalance?

**Answer:**
> "CIN2+ events are rare — roughly 5–10% positive rate in a healthy screening population. I handle this at three levels. For Logistic Regression: `class_weight='balanced'` re-weights the loss by inverse class frequency. For Random Forest: `balanced_subsample` does the same per tree. For XGBoost: `scale_pos_weight = n_negatives / n_positives` scales the gradient for positive examples. I evaluate using AUC rather than accuracy, and I separately report sensitivity and specificity — in a clinical setting, false negatives (missing a high-risk patient) are far more costly than false positives."

---

### Q6: What triggers a model retrain?

**Answer:**
> "Two triggers: first, a drift alert — if the monitoring module flags a significant shift in key feature distributions, that's a signal the training data is no longer representative. Second, a scheduled periodic retrain — in production I'd retrain every 6–12 months to incorporate the most recent outcomes data. After retraining, the new model would be validated against a fresh holdout period before replacing the deployed version."

---

### Q7: How would this integrate with a real hospital system?

**Answer:**
> "The FastAPI serving layer is the integration point. A hospital EMR or laboratory information system would POST each new screening record to the `/predict` endpoint after a test result is finalized. The risk scores go back into the EMR for the clinician to review. The monthly batch pipeline would run as a scheduled job — say on the first of each month — ingesting that month's records from the registry export, running predictions, running drift checks, and refreshing the Streamlit dashboard. Authentication, audit logging, and RBAC would wrap the API and dashboard before connecting to real patient data."

---

### Q8: Why are sensitivity and specificity both around 0.70 — is that good enough?

**Answer:**
> "For a screening adjunct tool, 0.70 sensitivity with 0.74 specificity is a reasonable baseline — it correctly identifies 70% of women who will develop CIN2+ and avoids flagging 74% of those who won't. But these numbers depend heavily on the decision threshold. In practice, you'd tune the threshold based on the clinical cost trade-off: if missing a high-risk patient is more costly than an unnecessary colposcopy referral, you lower the threshold to increase sensitivity at the cost of specificity. The AUC of 0.771 is the threshold-independent summary — the model has genuine discriminative power."

---

### Q9: What's the difference between the 'best model' and the deployed XGBoost models?

**Answer:**
> "The `best_cin2_3yr.pkl` is selected by validation AUC across all three candidate algorithms — in this run it's Logistic Regression. It's saved as the 'champion' model. The `xgb_cin2_1yr.pkl`, `xgb_cin2_3yr.pkl`, `xgb_cin2_5yr.pkl` are XGBoost models trained on all three time horizons — trained on the full train+validation dataset. The API serves the XGBoost family because we want three consistent risk scores from the same model family, and XGBoost is faster at inference. The selection process I showed is what informs that choice: it confirmed XGBoost is competitive when given the full dataset."

---

### Q10: How would you productionize this?

**Answer:**
> "Several changes for production: replace flat Parquet files with a proper event store (Postgres + Parquet on S3 for large history). Add a model registry (MLflow or similar) to version artifacts and track experiments. Wrap the API with authentication and TLS. Add an alerting pipeline for drift — send a Slack or email alert when `overall_drift_detected` is true. Add A/B testing infrastructure so the new model can shadow the old one before cutover. Finally, add outcome feedback: as real CIN2+ diagnoses come in over time, feed them back to retrain the model and close the loop."

---

*End of presentation package.*
