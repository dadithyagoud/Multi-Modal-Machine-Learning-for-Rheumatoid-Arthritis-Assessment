"""
02_preprocessing_a.py — Preprocessing: Metabolomics Dataset (Model A)
======================================================================
Source files (Hur et al. 2021, Arthritis Research & Therapy):
    data/MLR_hd4_higher_group_public.tsv  — 52 patients, high DAS28  (>=3.2)
    data/MLR_hd4_lower_group_public.tsv   — 76 patients, low  DAS28  (<3.2)

TASK: Binary classification
    Label=1 : High DAS28 activity (higher group, DAS28 >= 3.2)
    Label=0 : Low  DAS28 activity (lower  group, DAS28 <  3.2)

    Both Model A and Model B now output probabilities on [0,1].
    This enables meaningful late fusion.

Features: 686 metabolites only (sex, age, DAS28 dropped)

Steps:
    1. Load & parse raw TSV files
    2. Assign binary labels
    3. Verify no DAS28 overlap between groups
    4. Align on common metabolites
    5. Combine -> 128 patients x 686 metabolites + Label
    6. Save -> data/metabolomics_processed.csv

Run before 03_model_a.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

HIGHER_FILE    = os.path.join(DATA_DIR, "MLR.hd4.higher.group.public.tsv")
LOWER_FILE     = os.path.join(DATA_DIR, "MLR.hd4.lower.group.public.tsv")
PROCESSED_FILE = os.path.join(DATA_DIR, "metabolomics_processed.csv")

META_ROWS = ["patientID", "sex", "age", "DAS28"]

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

def parse_tsv(filepath, label, meta_rows):
    raw = pd.read_csv(filepath, sep="\t", index_col=0)

    # FIX 1: Warn if any expected meta rows are missing (catches naming mismatches
    # like "das28" vs "DAS28" that would cause silent data leakage into features)
    missing_meta = [r for r in meta_rows if r not in raw.index]
    if missing_meta:
        print(f"  WARNING: Expected meta rows not found in {filepath}: {missing_meta}")
        print(f"  Available index values (first 10): {list(raw.index[:10])}")

    das  = raw.loc["DAS28"].astype(float)
    mets = raw.drop(index=[r for r in meta_rows if r in raw.index]).astype(float).T
    mets.index.name = "sample"

    # FIX 2: Align DAS28 scores by patient ID (index), not by position.
    # Using .values directly assumes column order is preserved after .T —
    # explicit index alignment prevents silent score misassignment.
    mets["DAS28"] = das.loc[mets.index].values

    mets["Label"] = label
    return mets

# =============================================================================
# 1. LOAD & PARSE
# =============================================================================
section("1. Load & Parse Raw TSV Files")

df_higher = parse_tsv(HIGHER_FILE, label=1, meta_rows=META_ROWS)
df_lower  = parse_tsv(LOWER_FILE,  label=0, meta_rows=META_ROWS)

print(f"  Higher group (Label=1) : {df_higher.shape[0]} patients")
print(f"    DAS28 range          : {df_higher['DAS28'].min():.4f} — {df_higher['DAS28'].max():.4f}")
print(f"  Lower  group (Label=0) : {df_lower.shape[0]} patients")
print(f"    DAS28 range          : {df_lower['DAS28'].min():.4f} — {df_lower['DAS28'].max():.4f}")

# FIX 3: Check for missing values immediately after parsing
print(f"\n  Missing value check (post-parse):")
print(f"    Higher group : {df_higher.isnull().sum().sum()} missing values")
print(f"    Lower  group : {df_lower.isnull().sum().sum()} missing values")
if df_higher.isnull().sum().sum() > 0 or df_lower.isnull().sum().sum() > 0:
    print("  WARNING: Missing values detected in raw data — consider imputation before modelling.")

# =============================================================================
# 2. VERIFY GROUP SEPARATION
# =============================================================================
section("2. Verify Group Separation")

overlap = df_lower["DAS28"].max() > df_higher["DAS28"].min()
print(f"  Lower  max DAS28 : {df_lower['DAS28'].max():.4f}")
print(f"  Higher min DAS28 : {df_higher['DAS28'].min():.4f}")
print(f"  Overlap between groups : {overlap}")
assert not overlap, "ERROR: DAS28 groups overlap — check source files!"
print(f"  -> Clean binary separation confirmed")

# DAS28 distribution by group
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_lower["DAS28"],  bins=20, alpha=0.6, color="steelblue",
        label=f"Low DAS28  (Label=0, n={len(df_lower)})",  edgecolor="white")
ax.hist(df_higher["DAS28"], bins=20, alpha=0.6, color="tomato",
        label=f"High DAS28 (Label=1, n={len(df_higher)})", edgecolor="white")
ax.axvline(3.2, color="black", linestyle="--", lw=1.5, label="Split threshold (DAS28=3.2)")
ax.set_xlabel("DAS28 Score")
ax.set_ylabel("Patient Count")
ax.set_title("Dataset A — DAS28 Distribution by Group")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
save_fig("A_01_das28_by_group.png")

# =============================================================================
# 3. ALIGN ON COMMON METABOLITES
# =============================================================================
section("3. Align on Common Metabolites")

higher_mets = set(df_higher.columns) - {"DAS28", "Label"}
lower_mets  = set(df_lower.columns)  - {"DAS28", "Label"}
common_mets = sorted(higher_mets & lower_mets)

print(f"  Metabolites in higher only : {len(higher_mets - lower_mets)}")
print(f"  Metabolites in lower  only : {len(lower_mets - higher_mets)}")
print(f"  Common metabolites         : {len(common_mets)}  <- using these")

df_higher = df_higher[common_mets + ["DAS28", "Label"]]
df_lower  = df_lower[common_mets  + ["DAS28", "Label"]]

# =============================================================================
# 4. COMBINE & SAVE
# =============================================================================
section("4. Combine & Save")

df_all = pd.concat([df_higher, df_lower], axis=0).reset_index(drop=True)
df_all = df_all.drop(columns=["DAS28"])    # DAS28 was only for verification

feature_cols = [c for c in df_all.columns if c != "Label"]
n_high = (df_all["Label"] == 1).sum()
n_low  = (df_all["Label"] == 0).sum()

print(f"  Combined shape : {df_all.shape[0]} patients x {len(feature_cols)} metabolites + Label")
print(f"  Label=1 (High) : {n_high} patients")
print(f"  Label=0 (Low)  : {n_low}  patients")
print(f"  Missing values : {df_all.isnull().sum().sum()}")

if df_all.isnull().sum().sum() > 0:
    print("  WARNING: Missing values present in final dataset — consider imputation before modelling.")

# Class balance chart
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Low DAS28\n(Label=0)", "High DAS28\n(Label=1)"],
       [n_low, n_high], color=["steelblue", "tomato"], edgecolor="white", width=0.5)
ax.set_ylabel("Patient Count")
ax.set_title("Dataset A — Class Distribution")
ax.spines[["top", "right"]].set_visible(False)
for i, v in enumerate([n_low, n_high]):
    ax.text(i, v + 0.5, str(v), ha="center", fontsize=12, fontweight="bold")
save_fig("A_02_class_distribution.png")

df_all.to_csv(PROCESSED_FILE, index=False)
print(f"\n  Saved -> {PROCESSED_FILE}")

section("Preprocessing Complete — run 03_model_a.py next")