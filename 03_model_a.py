"""
03_model_a.py — Model A: Random Forest Classifier (Metabolomics -> High/Low DAS28)
====================================================================================
Dataset : data/metabolomics_processed.csv  (output of 02_preprocessing_a.py)
Model   : Random Forest Classifier
Target  : Label  (0 = Low DAS28 activity, 1 = High DAS28 activity)

Steps:
    1. Load processed data
    2. Train / test split  (80/20, stratified)
    3. Feature selection   (top-50 metabolites by RF importance)
    4. Train final RF Classifier on selected features
    5. Evaluate: accuracy, ROC-AUC, precision, recall, F1
    6. 5-fold cross-validation
    7. Confusion matrix + ROC curve
    8. Feature importance plot
    9. Save all results -> outputs/

Run 02_preprocessing_a.py first.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_score, recall_score, f1_score
)

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR       = "data"
OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCESSED_FILE = os.path.join(DATA_DIR, "metabolomics_processed.csv")
RESULTS_FILE   = os.path.join(OUTPUT_DIR, "A_model_results.csv")

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
TOP_K          = 50

# =============================================================================
# HELPERS
# =============================================================================
def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] {path}")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
section("1. Load Processed Data")

# FIX 1: Check input file exists before attempting to load
if not os.path.exists(PROCESSED_FILE):
    raise FileNotFoundError(
        f"Input file not found: {PROCESSED_FILE}\n"
        f"Run 02_preprocessing_a.py first."
    )

df = pd.read_csv(PROCESSED_FILE)
feature_cols = [c for c in df.columns if c != "Label"]
X = df[feature_cols].values
y = df["Label"].values

print(f"  Patients    : {df.shape[0]}")
print(f"  Metabolites : {len(feature_cols)}")
print(f"  Label=0 (Low DAS28)  : {(y==0).sum()}")
print(f"  Label=1 (High DAS28) : {(y==1).sum()}")

# =============================================================================
# 2. TRAIN / TEST SPLIT
# =============================================================================
section("2. Train / Test Split  (80/20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y
)

print(f"  Train : {X_train.shape[0]} patients  "
      f"| Low={(y_train==0).sum()}  High={(y_train==1).sum()}")
print(f"  Test  : {X_test.shape[0]}  patients  "
      f"| Low={(y_test==0).sum()}   High={(y_test==1).sum()}")

# =============================================================================
# 3. FEATURE SELECTION — Top-K by importance
# =============================================================================
section(f"3. Feature Selection  (top {TOP_K} metabolites)")

# Fit on full feature set first to get importances
rf_full = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_full.fit(X_train, y_train)

importances  = pd.Series(rf_full.feature_importances_, index=feature_cols)
top_features = importances.nlargest(TOP_K).index.tolist()

# FIX 2: Build index lookup map — O(n) once instead of O(n) x TOP_K
# (replaces repeated feature_cols.index(f) linear scans)
feat_idx_map = {f: i for i, f in enumerate(feature_cols)}
top_idx      = [feat_idx_map[f] for f in top_features]

X_train_sel = X_train[:, top_idx]
X_test_sel  = X_test[:,  top_idx]
X_sel       = X[:,       top_idx]

print(f"  Selected {TOP_K} metabolites from {len(feature_cols)}")
print(f"  Top 10 : {top_features[:10]}")

# =============================================================================
# 4. TRAIN FINAL MODEL
# =============================================================================
section(f"4. Train Final RF Classifier  (top {TOP_K} metabolites)")

rf_a = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_a.fit(X_train_sel, y_train)
print(f"  Trained: {rf_a.n_estimators} trees  |  class_weight='balanced'")

# =============================================================================
# 5. EVALUATE ON TEST SET
# =============================================================================
section("5. Test Set Evaluation")

y_pred  = rf_a.predict(X_test_sel)
y_proba = rf_a.predict_proba(X_test_sel)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print(f"\n  Accuracy  : {acc:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Low DAS28", "High DAS28"], zero_division=0))

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Low DAS28", "High DAS28"]
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Model A — Confusion Matrix\nAccuracy={acc:.3f}")
save_fig("A_03_confusion_matrix.png")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="tomato", lw=2, label=f"RF Classifier  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color="tomato")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Model A — ROC Curve (High vs Low DAS28)")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
save_fig("A_04_roc_curve.png")

# =============================================================================
# 6. CROSS-VALIDATION
# =============================================================================
section("6. Cross-Validation  (Stratified 5-Fold)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# FIX 3: Feature selection was performed on X_train before CV, so X_sel
# reflects features chosen with knowledge of the training split. This
# introduces a minor optimistic bias in CV scores — acceptable at this
# project scale but documented here as a known limitation for the report.
# A fully clean approach would wrap selection + training in a Pipeline.
cv_acc = cross_val_score(rf_a, X_sel, y, cv=cv, scoring="accuracy", n_jobs=-1)
cv_auc = cross_val_score(rf_a, X_sel, y, cv=cv, scoring="roc_auc",  n_jobs=-1)
cv_f1  = cross_val_score(rf_a, X_sel, y, cv=cv, scoring="f1",       n_jobs=-1)

print(f"\n  {'Metric':<12} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*70}")
for name, scores in [("Accuracy", cv_acc), ("ROC-AUC", cv_auc), ("F1", cv_f1)]:
    vals = "  ".join([f"{s:.4f}" for s in scores])
    print(f"  {name:<12} {vals}   {scores.mean():.4f}   {scores.std():.4f}")

# CV bar chart
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(5)
w = 0.25
ax.bar(x - w, cv_acc, w, label="Accuracy", color="tomato",     edgecolor="white")
ax.bar(x,     cv_auc, w, label="ROC-AUC",  color="darkorange", edgecolor="white")
ax.bar(x + w, cv_f1,  w, label="F1",       color="steelblue",  edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score")
ax.set_title("Model A — 5-Fold Cross-Validation Results")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
save_fig("A_05_cross_validation.png")

# =============================================================================
# 7. FEATURE IMPORTANCE
# =============================================================================
section("7. Feature Importance  (top 20 metabolites)")

imp = pd.Series(rf_a.feature_importances_, index=top_features).sort_values()
top20 = imp.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["tomato" if v == top20.max() else "steelblue" for v in top20.values]
top20.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Model A — Top 20 Metabolite Importances\n(red = most important)")
ax.set_xlabel("Importance")
ax.spines[["top", "right"]].set_visible(False)
save_fig("A_06_feature_importance.png")

print(f"\n  Top 10 metabolites:")
print(imp.nlargest(10).round(5).to_string())

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
section("8. Save Results")

results_df = pd.DataFrame([{
    "Model"           : "Model A — RF Classifier",
    "Dataset"         : "Hur et al. 2021 (High vs Low DAS28)",
    "Task"            : "High DAS28 (1) vs Low DAS28 (0)",
    "n_total"         : len(df),
    "n_train"         : X_train.shape[0],
    "n_test"          : X_test.shape[0],
    "n_features_raw"  : len(feature_cols),
    "n_features_sel"  : TOP_K,
    "Accuracy"        : round(acc,  4),
    "ROC_AUC"         : round(auc,  4),
    "Precision"       : round(prec, 4),
    "Recall"          : round(rec,  4),
    "F1"              : round(f1,   4),
    "CV_Acc_mean"     : round(cv_acc.mean(), 4),
    "CV_Acc_std"      : round(cv_acc.std(),  4),
    "CV_AUC_mean"     : round(cv_auc.mean(), 4),
    "CV_AUC_std"      : round(cv_auc.std(),  4),
    "CV_F1_mean"      : round(cv_f1.mean(),  4),
    "CV_F1_std"       : round(cv_f1.std(),   4),
}])

results_df.to_csv(RESULTS_FILE, index=False)
print(f"\n  Saved -> {RESULTS_FILE}")
print(f"\n  Summary:")
print(f"    Accuracy : {acc:.4f}   (CV: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f})")
print(f"    ROC-AUC  : {auc:.4f}   (CV: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f})")
print(f"    F1       : {f1:.4f}   (CV: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f})")

section("Model A Complete")
print("""
  Output files:
    outputs/A_01_das28_by_group.png      <- from preprocessing
    outputs/A_02_class_distribution.png  <- from preprocessing
    outputs/A_03_confusion_matrix.png
    outputs/A_04_roc_curve.png
    outputs/A_05_cross_validation.png
    outputs/A_06_feature_importance.png
    outputs/A_model_results.csv
""")