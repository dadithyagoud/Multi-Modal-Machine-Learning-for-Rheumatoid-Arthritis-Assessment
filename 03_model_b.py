"""
03_model_b.py — Model B: Random Forest Classifier (Clinical -> RA vs Normal)
=============================================================================
Dataset : data/clinical_processed.csv  (output of 02_preprocessing_b.py)
Model   : Random Forest Classifier
Target  : Label  (0 = Normal, 1 = Rheumatoid Arthritis)

Steps:
    1. Load processed data
    2. Train / test split  (stratified 80/20)
    3. Train Random Forest Classifier
    4. Evaluate: accuracy, ROC-AUC, precision, recall, F1, confusion matrix
    5. 5-fold cross-validation
    6. Feature importance plot
    7. ROC curve plot
    8. Save all results -> outputs/

Run 02_preprocessing_b.py first.
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

PROCESSED_FILE = os.path.join(DATA_DIR, "clinical_processed.csv")
RESULTS_FILE   = os.path.join(OUTPUT_DIR, "B_model_results.csv")

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
N_SPLITS       = 5

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
        f"Run 02_preprocessing_b.py first."
    )

df = pd.read_csv(PROCESSED_FILE)

feature_cols = [c for c in df.columns if c != "Label"]
X = df[feature_cols].values
y = df["Label"].values

print(f"  Shape          : {df.shape[0]} rows x {len(feature_cols)} features")
print(f"  Features       : {feature_cols}")
print(f"  Label balance  : Normal={(y==0).sum()}  RA={(y==1).sum()}")

# =============================================================================
# 2. TRAIN / TEST SPLIT
# =============================================================================
section("2. Train / Test Split  (80/20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  Train : {X_train.shape[0]} patients  "
      f"| Normal={(y_train==0).sum()}  RA={(y_train==1).sum()}")
print(f"  Test  : {X_test.shape[0]} patients   "
      f"| Normal={(y_test==0).sum()}   RA={(y_test==1).sum()}")

# =============================================================================
# 3. TRAIN RANDOM FOREST CLASSIFIER
# =============================================================================
section("3. Train Random Forest Classifier")

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",       # handles RA/Normal imbalance (2848 vs 1604)
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)
print(f"  Trained : {rf_clf.n_estimators} trees")
n_features_split = int(np.sqrt(len(feature_cols)))
print(f"  Features per split (sqrt of {len(feature_cols)}) : ~{n_features_split}")
print(f"  class_weight='balanced' applied to handle class imbalance")

# =============================================================================
# 4. EVALUATE ON TEST SET
# =============================================================================
section("4. Test Set Evaluation")

y_pred  = rf_clf.predict(X_test)
y_proba = rf_clf.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
# FIX 2: Added zero_division=0 to match model_a.py and prevent warnings
#         if the model predicts only one class on the test set
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred,    zero_division=0)
f1   = f1_score(y_test, y_pred,        zero_division=0)

print(f"\n  Accuracy  : {acc:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Precision : {prec:.4f}  (of predicted RA, how many truly RA)")
print(f"  Recall    : {rec:.4f}  (of all true RA, how many caught)")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "RA"],
                             zero_division=0))

# --- Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Normal", "RA"]
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Model B — Confusion Matrix\nAccuracy = {acc:.3f}")
save_fig("B_03_confusion_matrix.png")

# --- ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"RF Classifier  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "r--", lw=1, label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Model B — ROC Curve (RA vs Normal)")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
save_fig("B_04_roc_curve.png")

# =============================================================================
# 5. CROSS-VALIDATION
# =============================================================================
section("5. Cross-Validation  (Stratified 5-Fold)")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# FIX 3: cross_val_score re-fits rf_clf from scratch internally on each fold
#         using the full X, y — results are correct and independent of the
#         model fitted above on X_train only
cv_acc = cross_val_score(rf_clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
cv_auc = cross_val_score(rf_clf, X, y, cv=cv, scoring="roc_auc",  n_jobs=-1)
cv_f1  = cross_val_score(rf_clf, X, y, cv=cv, scoring="f1",       n_jobs=-1)

print(f"\n  {'Metric':<12} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*70}")
for name, scores in [("Accuracy", cv_acc), ("ROC-AUC", cv_auc), ("F1", cv_f1)]:
    vals = "  ".join([f"{s:.4f}" for s in scores])
    print(f"  {name:<12} {vals}   {scores.mean():.4f}   {scores.std():.4f}")

# --- CV bar chart
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(N_SPLITS)
w = 0.25
ax.bar(x - w,   cv_acc, w, label="Accuracy", color="steelblue",  edgecolor="white")
ax.bar(x,       cv_auc, w, label="ROC-AUC",  color="darkorange", edgecolor="white")
ax.bar(x + w,   cv_f1,  w, label="F1",       color="seagreen",   edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i+1}" for i in range(N_SPLITS)])
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score")
ax.set_title("Model B — 5-Fold Cross-Validation Results")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
save_fig("B_05_cross_validation.png")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
section("6. Feature Importance")

importances = pd.Series(rf_clf.feature_importances_, index=feature_cols)
importances_sorted = importances.sort_values(ascending=True)

print(f"\n  Feature importances (highest to lowest):")
print(importances.sort_values(ascending=False).round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["tomato" if v == importances.max() else "steelblue"
          for v in importances_sorted.values]
importances_sorted.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Model B — Feature Importances\n(red = most important)")
ax.set_xlabel("Importance")
ax.spines[["top", "right"]].set_visible(False)
save_fig("B_06_feature_importance.png")

# =============================================================================
# 7. SAVE RESULTS SUMMARY
# =============================================================================
section("7. Save Results")

results_df = pd.DataFrame([{
    "Model"           : "Model B — RF Classifier",
    "Dataset"         : "Mahdi et al. 2025 (RA vs Normal)",
    "n_total"         : len(df),
    "n_train"         : X_train.shape[0],
    "n_test"          : X_test.shape[0],
    "n_features"      : len(feature_cols),
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
print(f"    Accuracy  : {acc:.4f}   (CV: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f})")
print(f"    ROC-AUC   : {auc:.4f}   (CV: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f})")
print(f"    F1 Score  : {f1:.4f}   (CV: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f})")

section("Model B Complete")
print("""
  Output files:
    outputs/B_01_class_distribution.png   <- from preprocessing
    outputs/B_02_missing_values.png       <- from preprocessing
    outputs/B_03_confusion_matrix.png
    outputs/B_04_roc_curve.png
    outputs/B_05_cross_validation.png
    outputs/B_06_feature_importance.png
    outputs/B_model_results.csv
""")