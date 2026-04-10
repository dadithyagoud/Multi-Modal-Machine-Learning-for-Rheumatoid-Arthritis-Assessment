"""
02_preprocessing_b.py — Preprocessing: Clinical Dataset (Model B)
==================================================================
Dataset : Rheumatic and Autoimmune Disease Dataset.xlsx (Mahdi et al. 2025)
Shape   : 12,085 patients × 15 columns (14 features + Disease target)

Class:
    ClinicalPreprocessor
        load_raw_data()
        filter_ra_vs_normal()
        inspect_missing_values()
        impute_missing_values()
        encode_categoricals()
        save()
        run()                    ← runs all steps in order

Run before 03_model_b.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

try:
    import openpyxl
except ImportError:
    raise ImportError("openpyxl is required to read .xlsx files: pip install openpyxl")

from utils import Logger, FigureSaver


class ClinicalPreprocessor:
    """
    Loads, filters, imputes, encodes, and saves the clinical dataset.

    Parameters
    ----------
    data_dir   : directory containing the raw XLSX file
    output_dir : directory where plots and processed CSV are saved
    """

    TARGET_COL   = "Disease"
    RA_LABEL     = "Rheumatoid Arthritis"
    NORMAL_LABEL = "Normal"

    CONT_COLS = ["Age", "ESR", "CRP", "RF", "Anti-CCP", "C3", "C4"]
    CAT_COLS  = ["Gender", "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]
    BINARY_COLS = ["HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]

    def __init__(self, data_dir: str = "data", output_dir: str = "outputs"):
        self.data_dir   = data_dir
        self.output_dir = output_dir
        self.logger     = Logger()
        self.saver      = FigureSaver(output_dir)

        self.raw_file       = os.path.join(data_dir, "Rheumatic and Autoimmune Disease Dataset.xlsx")
        self.processed_file = os.path.join(data_dir, "clinical_processed.csv")

        self.df_raw:  pd.DataFrame | None = None
        self.df:      pd.DataFrame | None = None
        self.df_proc: pd.DataFrame | None = None

        os.makedirs(data_dir,   exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public pipeline steps
    # ------------------------------------------------------------------

    def load_raw_data(self) -> None:
        """Step 1 — Load the raw XLSX file."""
        self.logger.section("1. Load Raw Data")

        self.df_raw = pd.read_excel(self.raw_file)

        print(f"  Raw shape  : {self.df_raw.shape[0]} rows x {self.df_raw.shape[1]} columns")
        print(f"  Columns    : {list(self.df_raw.columns)}")
        print(f"\n  All disease classes:")
        print(self.df_raw[self.TARGET_COL].value_counts().to_string())

    def filter_ra_vs_normal(self) -> None:
        """Step 2 — Keep only RA and Normal rows; print class counts."""
        self.logger.section("2. Filter: RA vs Normal")

        self.df = self.df_raw[
            self.df_raw[self.TARGET_COL].isin([self.RA_LABEL, self.NORMAL_LABEL])
        ].copy().reset_index(drop=True)

        n_ra     = (self.df[self.TARGET_COL] == self.RA_LABEL).sum()
        n_normal = (self.df[self.TARGET_COL] == self.NORMAL_LABEL).sum()

        print(f"\n  Rheumatoid Arthritis : {n_ra}")
        print(f"  Normal               : {n_normal}")
        print(f"  Total after filter   : {len(self.df)}")
        print(f"\n  Class imbalance note : RA={n_ra} ({n_ra/len(self.df)*100:.1f}%), "
              f"Normal={n_normal} ({n_normal/len(self.df)*100:.1f}%)")
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
        self.saver.save("B_01_class_distribution.png")

    def inspect_missing_values(self) -> None:
        """Step 3 — Report per-column missing value counts and plot."""
        self.logger.section("3. Missing Value Analysis")

        missing_count = self.df.isnull().sum()
        missing_pct   = (missing_count / len(self.df) * 100).round(2)
        miss_df = pd.DataFrame({
            "Missing Count": missing_count,
            "Missing %"    : missing_pct
        }).sort_values("Missing %", ascending=False)

        print(f"\n  Total missing cells : {self.df.isnull().sum().sum()}")
        print(f"\n  Per-column breakdown:")
        print(miss_df[miss_df["Missing Count"] > 0].to_string())

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
            self.saver.save("B_02_missing_values.png")
        else:
            print("  No missing values found — skipping missing values plot.")

    def impute_missing_values(self) -> None:
        """Step 4 — Median imputation for continuous, mode for categorical."""
        self.logger.section("4. Imputation")

        self.df_proc = self.df.copy()

        imp_cont = SimpleImputer(strategy="median")
        self.df_proc[self.CONT_COLS] = imp_cont.fit_transform(self.df_proc[self.CONT_COLS])
        print(f"  Continuous  ({len(self.CONT_COLS)} cols) : median imputation")
        for col in self.CONT_COLS:
            print(f"    {col:<12} median = {self.df[col].median():.2f}")

        imp_cat = SimpleImputer(strategy="most_frequent")
        self.df_proc[self.CAT_COLS] = imp_cat.fit_transform(self.df_proc[self.CAT_COLS])
        print(f"\n  Categorical ({len(self.CAT_COLS)} cols) : most-frequent imputation")
        for col in self.CAT_COLS:
            print(f"    {col:<14} most frequent = {self.df[col].mode()[0]}")

        print(f"\n  Missing values after imputation : {self.df_proc.isnull().sum().sum()}")

    def encode_categoricals(self) -> None:
        """Step 5 — Encode Gender and Positive/Negative columns to 0/1."""
        self.logger.section("5. Encoding")

        self.df_proc["Gender"] = self.df_proc["Gender"].map({"Male": 0, "Female": 1})
        print("  Gender         : Male=0, Female=1")

        for col in self.BINARY_COLS:
            self.df_proc[col] = self.df_proc[col].map({"Positive": 1, "Negative": 0})
        print(f"  Binary cols    : Positive=1, Negative=0 -> {self.BINARY_COLS}")

        self.df_proc["Label"] = self.df_proc[self.TARGET_COL].map(
            {self.NORMAL_LABEL: 0, self.RA_LABEL: 1}
        )
        self.df_proc = self.df_proc.drop(columns=[self.TARGET_COL])
        print(f"  Target         : Normal=0, RA=1  ->  stored as 'Label'")

        post_enc_nulls = self.df_proc[["Gender"] + self.BINARY_COLS].isnull().sum()
        if post_enc_nulls.sum() > 0:
            print("\n  WARNING: Unexpected values found after encoding — NaN introduced!")
            print(post_enc_nulls[post_enc_nulls > 0].to_string())
        else:
            print("  Encoding check passed — no NaN introduced.")

    def save(self) -> None:
        """Step 6 — Print final stats and save processed CSV."""
        self.logger.section("6. Final Dataset")

        feature_cols = [c for c in self.df_proc.columns if c != "Label"]

        print(f"\n  Shape          : {self.df_proc.shape[0]} rows x {self.df_proc.shape[1]} columns")
        print(f"  Features ({len(feature_cols)}) : {feature_cols}")
        print(f"  Label balance  : Normal={(self.df_proc['Label']==0).sum()}  RA={(self.df_proc['Label']==1).sum()}")
        print(f"  Missing values : {self.df_proc.isnull().sum().sum()}")

        print(f"\n  Continuous feature statistics (post-imputation):")
        print(self.df_proc[self.CONT_COLS].describe().round(2).to_string())

        print(f"\n  Categorical feature value counts (post-encoding):")
        for col in ["Gender"] + self.BINARY_COLS:
            vc = self.df_proc[col].value_counts().to_dict()
            print(f"    {col:<14}: {vc}")

        self.df_proc.to_csv(self.processed_file, index=False)
        print(f"\n  Saved -> {self.processed_file}")

    def run(self) -> None:
        """Run all preprocessing steps in order."""
        self.load_raw_data()
        self.filter_ra_vs_normal()
        self.inspect_missing_values()
        self.impute_missing_values()
        self.encode_categoricals()
        self.save()
        self.logger.section("Preprocessing Complete — run 03_model_b.py next")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    preprocessor = ClinicalPreprocessor(data_dir="data", output_dir="outputs")
    preprocessor.run()
