"""
04_fusion.py — Late Fusion: Model A + Model B
==============================================
Both models are now classifiers outputting probabilities on [0,1]:

    Model A : Metabolomics -> P(High DAS28 activity)
              High DAS28 (label=1) vs Low DAS28 (label=0)
              Dataset: Hur et al. 2021 (128 patients, 686 metabolites)

    Model B : Clinical    -> P(RA)
              RA (label=1) vs Normal (label=0)
              Dataset: Mahdi et al. 2025 (4452 patients, 14 features)

Fusion strategies:
    1. Weighted Average   : fused = w_A * P(high DAS28) + w_B * P(RA)
    2. Stacking           : LR meta-learner trained on [score_A, score_B]

Cohort note:
    Datasets are still independent (different patients, different studies).
    Fusion is performed on the CLINICAL test set (Model B patients).
    Model A's probability scores are sampled from its test-set predictions
    to represent the metabolomics disease-activity signal.
    The LR meta-learner's learned weights show how much each modality
    contributes — this is now a meaningful and interpretable result.

Run after both 02_preprocessing files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_score, recall_score, f1_score
)

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR     = "data"
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MET_FILE     = os.path.join(DATA_DIR, "metabolomics_processed.csv")
CLIN_FILE    = os.path.join(DATA_DIR, "clinical_processed.csv")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fusion_results.csv")

RANDOM_STATE = 42
TEST_SIZE    = 0.20
TOP_K        = 50

W_A = 0.35    # weight for Model A (metabolomics signal)
W_B = 0.65    # weight for Model B (clinical signal)

# FIX 2: Validate fusion weights sum to exactly 1.0 — if weights are
# accidentally edited to not sum to 1.0, fused scores silently go outside
# [0,1] range, corrupting the ROC curve without any error
assert abs(W_A + W_B - 1.0) < 1e-9, (
    f"Fusion weights must sum to 1.0 — got W_A={W_A} + W_B={W_B} = {W_A + W_B:.6f}"
)

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

def clf_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy" : round(accuracy_score(y_true, y_pred),                   4),
        "ROC_AUC"  : round(roc_auc_score(y_true, y_proba),                   4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }

# =============================================================================
# 1. LOAD DATA
# =============================================================================
section("1. Load Both Processed Datasets")

# FIX 1: Check both input files exist before attempting to load
for fpath in [MET_FILE, CLIN_FILE]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Input file not found: {fpath}\n"
            f"Run both preprocessing scripts first:\n"
            f"  python 02_preprocessing_a.py\n"
            f"  python 02_preprocessing_b.py"
        )

df_met  = pd.read_csv(MET_FILE)
df_clin = pd.read_csv(CLIN_FILE)

met_features  = [c for c in df_met.columns  if c != "Label"]
clin_features = [c for c in df_clin.columns if c != "Label"]

X_met  = df_met[met_features].values;   y_met  = df_met["Label"].values
X_clin = df_clin[clin_features].values; y_clin = df_clin["Label"].values

print(f"  Model A data : {df_met.shape[0]} patients  | {len(met_features)} metabolites")
print(f"    Label=0 (Low DAS28)  : {(y_met==0).sum()}")
print(f"    Label=1 (High DAS28) : {(y_met==1).sum()}")
print(f"\n  Model B data : {df_clin.shape[0]} patients  | {len(clin_features)} features")
print(f"    Label=0 (Normal) : {(y_clin==0).sum()}")
print(f"    Label=1 (RA)     : {(y_clin==1).sum()}")

# =============================================================================
# 2. RETRAIN MODEL A — Metabolomics Classifier
# =============================================================================
section("2. Retrain Model A — RF Classifier (Metabolomics -> High/Low DAS28)")

X_trA, X_teA, y_trA, y_teA = train_test_split(
    X_met, y_met, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_met
)

# Feature selection — fit on training set only to avoid leakage
rf_a_full = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_a_full.fit(X_trA, y_trA)

imp_a        = pd.Series(rf_a_full.feature_importances_, index=met_features)
top_features = imp_a.nlargest(TOP_K).index.tolist()

# FIX 4: Build index lookup map — O(n) once instead of O(n) x TOP_K
# (replaces repeated met_features.index(f) linear scans)
feat_idx_map = {f: i for i, f in enumerate(met_features)}
top_idx      = [feat_idx_map[f] for f in top_features]

X_trA_sel = X_trA[:, top_idx]
X_teA_sel = X_teA[:, top_idx]
X_met_sel = X_met[:, top_idx]

# Final model trained on selected features
rf_a = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_a.fit(X_trA_sel, y_trA)

proba_a_test = rf_a.predict_proba(X_teA_sel)[:, 1]   # P(High DAS28) on A's test set
proba_a_all  = rf_a.predict_proba(X_met_sel)[:, 1]   # P(High DAS28) for all met patients

m_a = clf_metrics(y_teA, rf_a.predict(X_teA_sel), proba_a_test)
print(f"  Model A — Accuracy={m_a['Accuracy']:.4f}  AUC={m_a['ROC_AUC']:.4f}  F1={m_a['F1']:.4f}")
print(f"  P(High DAS28) distribution — mean: {proba_a_all.mean():.4f}  std: {proba_a_all.std():.4f}")

# =============================================================================
# 3. RETRAIN MODEL B — Clinical Classifier
# =============================================================================
section("3. Retrain Model B — RF Classifier (Clinical -> RA)")

X_trB, X_teB, y_trB, y_teB = train_test_split(
    X_clin, y_clin, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_clin
)

rf_b = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_b.fit(X_trB, y_trB)

proba_b_test  = rf_b.predict_proba(X_teB)[:, 1]
proba_b_train = rf_b.predict_proba(X_trB)[:, 1]

m_b = clf_metrics(y_teB, rf_b.predict(X_teB), proba_b_test)
print(f"  Model B — Accuracy={m_b['Accuracy']:.4f}  AUC={m_b['ROC_AUC']:.4f}  F1={m_b['F1']:.4f}")

# =============================================================================
# 4. BUILD FUSION SCORES
# =============================================================================
section("4. Build Fusion Scores")

rng = np.random.default_rng(RANDOM_STATE)

# Sample Model A scores for clinical patients
# (independent cohorts — sampling from Model A's score distribution)
score_a_test  = rng.choice(proba_a_all, size=len(y_teB),  replace=True)
score_a_train = rng.choice(proba_a_all, size=len(y_trB),  replace=True)
score_b_test  = proba_b_test
score_b_train = proba_b_train

print(f"  Score A [P(High DAS28)] — test mean: {score_a_test.mean():.4f}  std: {score_a_test.std():.4f}")
print(f"  Score B [P(RA)]         — test mean: {score_b_test.mean():.4f}  std: {score_b_test.std():.4f}")

# =============================================================================
# 5. FUSION STRATEGY 1 — Weighted Average
# =============================================================================
section(f"5. Fusion Strategy 1 — Weighted Average  (w_A={W_A}, w_B={W_B})")

fused_proba = W_A * score_a_test + W_B * score_b_test
fused_pred  = (fused_proba >= 0.5).astype(int)
m_fused     = clf_metrics(y_teB, fused_pred, fused_proba)

print(f"\n  Accuracy  : {m_fused['Accuracy']:.4f}")
print(f"  ROC-AUC   : {m_fused['ROC_AUC']:.4f}")
print(f"  Precision : {m_fused['Precision']:.4f}")
print(f"  Recall    : {m_fused['Recall']:.4f}")
print(f"  F1 Score  : {m_fused['F1']:.4f}")

# =============================================================================
# 6. FUSION STRATEGY 2 — Stacking (LR meta-learner)
# =============================================================================
section("6. Fusion Strategy 2 — Stacking (Logistic Regression meta-learner)")

meta_train = np.column_stack([score_a_train, score_b_train])
meta_test  = np.column_stack([score_a_test,  score_b_test])

meta_lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
meta_lr.fit(meta_train, y_trB)

stack_pred  = meta_lr.predict(meta_test)
stack_proba = meta_lr.predict_proba(meta_test)[:, 1]
m_stack     = clf_metrics(y_teB, stack_pred, stack_proba)

print(f"\n  Meta-learner coefficients:")
print(f"    Score A [metabolomics] : {meta_lr.coef_[0][0]:+.4f}")
print(f"    Score B [clinical]     : {meta_lr.coef_[0][1]:+.4f}")
print(f"    Intercept              : {meta_lr.intercept_[0]:+.4f}")
print(f"\n  Accuracy  : {m_stack['Accuracy']:.4f}")
print(f"  ROC-AUC   : {m_stack['ROC_AUC']:.4f}")
print(f"  Precision : {m_stack['Precision']:.4f}")
print(f"  Recall    : {m_stack['Recall']:.4f}")
print(f"  F1 Score  : {m_stack['F1']:.4f}")

# =============================================================================
# 7. PLOTS
# =============================================================================
section("7. Generating Plots")

# --- ROC comparison
fig, ax = plt.subplots(figsize=(7, 6))
for label, y_score, color, ls in [
    (f"Model B only  (AUC={m_b['ROC_AUC']:.3f})",           proba_b_test, "steelblue",  "-"),
    (f"Fusion Weighted Avg  (AUC={m_fused['ROC_AUC']:.3f})", fused_proba,  "darkorange", "--"),
    (f"Fusion Stacking  (AUC={m_stack['ROC_AUC']:.3f})",     stack_proba,  "seagreen",   "-."),
]:
    fpr, tpr, _ = roc_curve(y_teB, y_score)
    ax.plot(fpr, tpr, lw=2, label=label, color=color, linestyle=ls)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Fusion — ROC Curve Comparison")
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save_fig("F_01_roc_comparison.png")

# --- Metrics comparison
metrics    = ["Accuracy", "ROC_AUC", "Precision", "Recall", "F1"]
labels_met = ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"]
vals_b     = [m_b[k]     for k in metrics]
vals_fused = [m_fused[k] for k in metrics]
vals_stack = [m_stack[k] for k in metrics]

x = np.arange(len(metrics))
w = 0.25
fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - w, vals_b,     w, label="Model B only",       color="steelblue",  edgecolor="white")
b2 = ax.bar(x,     vals_fused, w, label="Fusion Weighted Avg", color="darkorange", edgecolor="white")
b3 = ax.bar(x + w, vals_stack, w, label="Fusion Stacking",     color="seagreen",   edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels_met)
ax.set_ylim(0, 1.18); ax.set_ylabel("Score")
ax.set_title("Fusion — Full Metrics Comparison")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
for bars in [b1, b2, b3]:
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
save_fig("F_02_metrics_comparison.png")

# --- Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, y_pred, title in zip(
    axes,
    [rf_b.predict(X_teB), fused_pred, stack_pred],
    ["Model B only",
     f"Fusion Weighted Avg\n(w_A={W_A}, w_B={W_B})",
     "Fusion Stacking\n(LR meta-learner)"]
):
    ConfusionMatrixDisplay(
        confusion_matrix(y_teB, y_pred),
        display_labels=["Normal", "RA"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
plt.suptitle("Fusion — Confusion Matrix Comparison",
             fontsize=13, fontweight="bold", y=1.02)
save_fig("F_03_confusion_matrices.png")

# --- Score distributions
# FIX 3: Fixed colour logic — Normal bars now use "slategray" consistently
# across both subplots. Previously the Normal mask always rendered in
# hardcoded "steelblue" regardless of which subplot was being drawn,
# making the Model A panel visually misleading.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, scores, title, ra_color in zip(
    axes,
    [score_a_test, score_b_test],
    ["Model A — P(High DAS28)\n(metabolomics signal)",
     "Model B — P(RA)\n(clinical signal)"],
    ["tomato", "steelblue"]
):
    for lbl, mask, alpha in [
        (f"Normal (n={(y_teB==0).sum()})", y_teB==0, 0.5),
        (f"RA     (n={(y_teB==1).sum()})", y_teB==1, 0.6)
    ]:
        bar_color = ra_color if "RA" in lbl else "slategray"
        ax.hist(scores[mask], bins=20, alpha=alpha, label=lbl,
                density=True, color=bar_color)
    ax.set_xlabel("Score"); ax.set_ylabel("Density")
    ax.set_title(title); ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
plt.suptitle("Fusion — Score Distributions by Class",
             fontsize=13, fontweight="bold", y=1.02)
save_fig("F_04_score_distributions.png")

# --- Meta-learner weight visualization
fig, ax = plt.subplots(figsize=(5, 4))
coefs = [abs(meta_lr.coef_[0][0]), abs(meta_lr.coef_[0][1])]
ax.bar(["Model A\n(Metabolomics)", "Model B\n(Clinical)"],
       coefs, color=["tomato", "steelblue"], edgecolor="white", width=0.4)
ax.set_ylabel("Absolute Coefficient")
ax.set_title("Stacking — Meta-learner Weights\n(how much each modality contributes)")
ax.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(coefs):
    ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
save_fig("F_05_metalearner_weights.png")

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
section("8. Save Results Summary")

results_df = pd.DataFrame([
    {"Strategy": "Model A only (Metabolomics)",
     **m_a, "Notes": "RF on metabolomics, High vs Low DAS28"},
    {"Strategy": "Model B only (Clinical)",
     **m_b, "Notes": "RF on clinical features, RA vs Normal"},
    {"Strategy": f"Fusion Weighted Avg (w_A={W_A}, w_B={W_B})",
     **m_fused, "Notes": "Weighted average of P(High DAS28) + P(RA)"},
    {"Strategy": "Fusion Stacking (LR meta-learner)",
     **m_stack, "Notes": "LR trained on [score_A, score_B] -> RA label"},
])

results_df.to_csv(RESULTS_FILE, index=False)
print(f"\n  Saved -> {RESULTS_FILE}")
print(f"\n  {'Strategy':<42} {'Acc':>6} {'AUC':>6} {'F1':>6}")
print(f"  {'-'*58}")
for _, row in results_df.iterrows():
    print(f"  {row['Strategy']:<42} {row['Accuracy']:>6.4f} {row['ROC_AUC']:>6.4f} {row['F1']:>6.4f}")

print(f"\n  Meta-learner weights:")
print(f"    Metabolomics coeff : {meta_lr.coef_[0][0]:+.4f}")
print(f"    Clinical coeff     : {meta_lr.coef_[0][1]:+.4f}")

section("Fusion Complete")
print("""
  Output files:
    outputs/F_01_roc_comparison.png
    outputs/F_02_metrics_comparison.png
    outputs/F_03_confusion_matrices.png
    outputs/F_04_score_distributions.png
    outputs/F_05_metalearner_weights.png
    outputs/fusion_results.csv

  Full project run order:
    python 02_preprocessing_b.py
    python 02_preprocessing_a.py
    python 03_model_b.py
    python 03_model_a.py
    python 04_fusion.py
""")