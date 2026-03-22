"""
02_preprocessing_b.py — Preprocessing: Clinical Dataset (Model B)
==================================================================
Dataset : Rheumatic and Autoimmune Disease Dataset.xlsx (Mahdi et al. 2025)
Shape   : 12,085 patients × 15 columns (14 features + Disease target)

Features:
    Continuous  : Age, ESR, CRP, RF, Anti-CCP, C3, C4
    Categorical : Gender, HLA-B27, ANA, Anti-Ro, Anti-La, Anti-dsDNA, Anti-Sm

Steps:
    1. Load raw xlsx
    2. Filter to RA vs Normal only  (2848 RA + 1604 Normal = 4452 patients)
    3. Inspect missing values
    4. Impute missing values
    5. Encode categorical columns
    6. Save cleaned dataset → data/clinical_processed.csv

Run before 03_model_b.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# FIX 1: Guard for openpyxl dependency required by pd.read_excel()
try:
    import openpyxl
except ImportError:
    raise ImportError("openpyxl is required to read .xlsx files: pip install openpyxl")

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RAW_FILE       = os.path.join(DATA_DIR, "Rheumatic and Autoimmune Disease Dataset.xlsx")
PROCESSED_FILE = os.path.join(DATA_DIR, "clinical_processed.csv")

TARGET_COL   = "Disease"
RA_LABEL     = "Rheumatoid Arthritis"
NORMAL_LABEL = "Normal"

# Continuous and categorical features (confirmed from dataset inspection)
CONT_COLS = ["Age", "ESR", "CRP", "RF", "Anti-CCP", "C3", "C4"]
CAT_COLS  = ["Gender", "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]

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
# 1. LOAD RAW DATA
# =============================================================================
section("1. Load Raw Data")

df_raw = pd.read_excel(RAW_FILE)

print(f"  Raw shape  : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
print(f"  Columns    : {list(df_raw.columns)}")
print(f"\n  All disease classes:")
print(df_raw[TARGET_COL].value_counts().to_string())

# =============================================================================
# 2. FILTER — RA vs Normal only
# =============================================================================
section("2. Filter: RA vs Normal")

df = df_raw[df_raw[TARGET_COL].isin([RA_LABEL, NORMAL_LABEL])].copy()
df = df.reset_index(drop=True)

n_ra     = (df[TARGET_COL] == RA_LABEL).sum()
n_normal = (df[TARGET_COL] == NORMAL_LABEL).sum()

print(f"\n  Rheumatoid Arthritis : {n_ra}")
print(f"  Normal               : {n_normal}")
print(f"  Total after filter   : {len(df)}")
print(f"\n  Class imbalance note : RA={n_ra} ({n_ra/len(df)*100:.1f}%), "
      f"Normal={n_normal} ({n_normal/len(df)*100:.1f}%)")
print(f"  -> Using class_weight='balanced' in RF to handle this.")

# Class balance bar chart
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Normal (0)", "RA (1)"], [n_normal, n_ra],
       color=["steelblue", "tomato"], edgecolor="white", width=0.5)
ax.set_ylabel("Patient Count")
ax.set_title("Model B — Class Distribution (RA vs Normal)")
ax.spines[["top", "right"]].set_visible(False)
for i, v in enumerate([n_normal, n_ra]):
    ax.text(i, v + 20, str(v), ha="center", fontsize=11, fontweight="bold")
save_fig("B_01_class_distribution.png")

# =============================================================================
# 3. INSPECT MISSING VALUES
# =============================================================================
section("3. Missing Value Analysis")

missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)
miss_df = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing %"    : missing_pct
}).sort_values("Missing %", ascending=False)

print(f"\n  Total missing cells : {df.isnull().sum().sum()}")
print(f"\n  Per-column breakdown:")
print(miss_df[miss_df["Missing Count"] > 0].to_string())

# FIX 2: Guard against empty plot when no missing values exist
miss_plot = miss_df[miss_df["Missing %"] > 0]
if not miss_plot.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(miss_plot.index, miss_plot["Missing %"],
                  color="tomato", edgecolor="white")
    ax.set_title("Model B — Missing Values per Column (RA + Normal subset)")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("Column")
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    save_fig("B_02_missing_values.png")
else:
    print("  No missing values found — skipping missing values plot.")

# =============================================================================
# 4. IMPUTE MISSING VALUES
# =============================================================================
section("4. Imputation")

df_proc = df.copy()

# Continuous -> median imputation (robust to outliers)
imp_cont = SimpleImputer(strategy="median")
df_proc[CONT_COLS] = imp_cont.fit_transform(df_proc[CONT_COLS])
print(f"  Continuous  ({len(CONT_COLS)} cols) : median imputation")
for col in CONT_COLS:
    print(f"    {col:<12} median = {df[col].median():.2f}")

# Categorical -> most frequent imputation
imp_cat = SimpleImputer(strategy="most_frequent")
df_proc[CAT_COLS] = imp_cat.fit_transform(df_proc[CAT_COLS])
print(f"\n  Categorical ({len(CAT_COLS)} cols) : most-frequent imputation")
for col in CAT_COLS:
    print(f"    {col:<14} most frequent = {df[col].mode()[0]}")

print(f"\n  Missing values after imputation : {df_proc.isnull().sum().sum()}")

# =============================================================================
# 5. ENCODE CATEGORICAL COLUMNS
# =============================================================================
section("5. Encoding")

# Gender: Male=0, Female=1
df_proc["Gender"] = df_proc["Gender"].map({"Male": 0, "Female": 1})
print("  Gender         : Male=0, Female=1")

# All Positive/Negative columns -> 1/0
binary_cols = ["HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]
for col in binary_cols:
    df_proc[col] = df_proc[col].map({"Positive": 1, "Negative": 0})
print(f"  Binary cols    : Positive=1, Negative=0 -> {binary_cols}")

# Target: Normal=0, RA=1
df_proc["Label"] = df_proc[TARGET_COL].map({NORMAL_LABEL: 0, RA_LABEL: 1})
df_proc = df_proc.drop(columns=[TARGET_COL])
print(f"  Target         : Normal=0, RA=1  ->  stored as 'Label'")

# FIX 3: Check for NaN introduced by encoding (catches unexpected value formats)
post_enc_nulls = df_proc[["Gender"] + binary_cols].isnull().sum()
if post_enc_nulls.sum() > 0:
    print("\n  WARNING: Unexpected values found after encoding — NaN introduced!")
    print(post_enc_nulls[post_enc_nulls > 0].to_string())
    print("  Check the raw data for unexpected category values (e.g. 'male' vs 'Male').")
else:
    print("  Encoding check passed — no NaN introduced.")

# =============================================================================
# 6. FINAL CHECK & SAVE
# =============================================================================
section("6. Final Dataset")

feature_cols = [c for c in df_proc.columns if c != "Label"]

print(f"\n  Shape          : {df_proc.shape[0]} rows x {df_proc.shape[1]} columns")
print(f"  Features ({len(feature_cols)}) : {feature_cols}")
print(f"  Label balance  : Normal={(df_proc['Label']==0).sum()}  RA={(df_proc['Label']==1).sum()}")
print(f"  Missing values : {df_proc.isnull().sum().sum()}")

print(f"\n  Continuous feature statistics (post-imputation):")
print(df_proc[CONT_COLS].describe().round(2).to_string())

print(f"\n  Categorical feature value counts (post-encoding):")
for col in ["Gender"] + binary_cols:
    vc = df_proc[col].value_counts().to_dict()
    print(f"    {col:<14}: {vc}")

df_proc.to_csv(PROCESSED_FILE, index=False)
print(f"\n  Saved -> {PROCESSED_FILE}")

section("Preprocessing Complete — run 03_model_b.py next")