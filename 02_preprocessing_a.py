"""
02_preprocessing_a.py — Preprocessing: Metabolomics Dataset (Model A)
======================================================================
Source files (Hur et al. 2021, Arthritis Research & Therapy):
    data/MLR_hd4_higher_group_public.tsv  — 52 patients, high DAS28 (>=3.2)
    data/MLR_hd4_lower_group_public.tsv   — 76 patients, low  DAS28 (<3.2)

Class:
    MetabolomicsPreprocessor
        load_and_parse()
        verify_group_separation()
        align_common_metabolites()
        combine_and_save()
        run()                    ← runs all steps in order

Run before 03_model_a.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import Logger, FigureSaver


class MetabolomicsPreprocessor:
    """
    Loads, validates, aligns, and saves the metabolomics dataset.

    Parameters
    ----------
    data_dir   : directory containing the raw TSV files
    output_dir : directory where plots and processed CSV are saved
    """

    META_ROWS = ["patientID", "sex", "age", "DAS28"]

    def __init__(self, data_dir: str = "data", output_dir: str = "outputs"):
        self.data_dir   = data_dir
        self.output_dir = output_dir
        self.logger     = Logger()
        self.saver      = FigureSaver(output_dir)

        self.higher_file    = os.path.join(data_dir, "MLR.hd4.higher.group.public.tsv")
        self.lower_file     = os.path.join(data_dir, "MLR.hd4.lower.group.public.tsv")
        self.processed_file = os.path.join(data_dir, "metabolomics_processed.csv")

        self.df_higher: pd.DataFrame | None = None
        self.df_lower:  pd.DataFrame | None = None
        self.df_all:    pd.DataFrame | None = None

        os.makedirs(data_dir,   exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_tsv(self, filepath: str, label: int) -> pd.DataFrame:
        """
        Reads one raw TSV file and returns a patient-row DataFrame
        with metabolite features, DAS28, and Label columns.
        """
        raw = pd.read_csv(filepath, sep="\t", index_col=0)

        missing_meta = [r for r in self.META_ROWS if r not in raw.index]
        if missing_meta:
            print(f"  WARNING: Expected meta rows not found in {filepath}: {missing_meta}")
            print(f"  Available index values (first 10): {list(raw.index[:10])}")

        das  = raw.loc["DAS28"].astype(float)
        mets = raw.drop(index=[r for r in self.META_ROWS if r in raw.index]).astype(float).T
        mets.index.name = "sample"

        # Align DAS28 by patient ID index, not by position
        mets["DAS28"] = das.loc[mets.index].values
        mets["Label"] = label
        return mets

    def _plot_das28_distribution(self) -> None:
        """Plots DAS28 score distribution for both groups."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self.df_lower["DAS28"],  bins=20, alpha=0.6, color="steelblue",
                label=f"Low DAS28  (Label=0, n={len(self.df_lower)})",  edgecolor="white")
        ax.hist(self.df_higher["DAS28"], bins=20, alpha=0.6, color="tomato",
                label=f"High DAS28 (Label=1, n={len(self.df_higher)})", edgecolor="white")
        ax.axvline(3.2, color="black", linestyle="--", lw=1.5, label="Split threshold (DAS28=3.2)")
        ax.set_xlabel("DAS28 Score")
        ax.set_ylabel("Patient Count")
        ax.set_title("Dataset A — DAS28 Distribution by Group")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        self.saver.save("A_01_das28_by_group.png")

    def _plot_class_distribution(self) -> None:
        """Plots class label counts after combining both groups."""
        n_low  = (self.df_all["Label"] == 0).sum()
        n_high = (self.df_all["Label"] == 1).sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Low DAS28\n(Label=0)", "High DAS28\n(Label=1)"],
               [n_low, n_high], color=["steelblue", "tomato"],
               edgecolor="white", width=0.5)
        ax.set_ylabel("Patient Count")
        ax.set_title("Dataset A — Class Distribution")
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate([n_low, n_high]):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=12, fontweight="bold")
        self.saver.save("A_02_class_distribution.png")

    # ------------------------------------------------------------------
    # Public pipeline steps
    # ------------------------------------------------------------------

    def load_and_parse(self) -> None:
        """Step 1 — Load and parse both raw TSV files."""
        self.logger.section("1. Load & Parse Raw TSV Files")

        self.df_higher = self._parse_tsv(self.higher_file, label=1)
        self.df_lower  = self._parse_tsv(self.lower_file,  label=0)

        print(f"  Higher group (Label=1) : {self.df_higher.shape[0]} patients")
        print(f"    DAS28 range          : {self.df_higher['DAS28'].min():.4f} — {self.df_higher['DAS28'].max():.4f}")
        print(f"  Lower  group (Label=0) : {self.df_lower.shape[0]} patients")
        print(f"    DAS28 range          : {self.df_lower['DAS28'].min():.4f} — {self.df_lower['DAS28'].max():.4f}")

        print(f"\n  Missing value check (post-parse):")
        print(f"    Higher group : {self.df_higher.isnull().sum().sum()} missing values")
        print(f"    Lower  group : {self.df_lower.isnull().sum().sum()} missing values")
        if self.df_higher.isnull().sum().sum() > 0 or self.df_lower.isnull().sum().sum() > 0:
            print("  WARNING: Missing values detected — consider imputation before modelling.")

    def verify_group_separation(self) -> None:
        """Step 2 — Assert DAS28 ranges do not overlap between groups."""
        self.logger.section("2. Verify Group Separation")

        overlap = self.df_lower["DAS28"].max() > self.df_higher["DAS28"].min()
        print(f"  Lower  max DAS28 : {self.df_lower['DAS28'].max():.4f}")
        print(f"  Higher min DAS28 : {self.df_higher['DAS28'].min():.4f}")
        print(f"  Overlap between groups : {overlap}")
        assert not overlap, "ERROR: DAS28 groups overlap — check source files!"
        print(f"  -> Clean binary separation confirmed")

        self._plot_das28_distribution()

    def align_common_metabolites(self) -> None:
        """Step 3 — Keep only metabolites present in both groups."""
        self.logger.section("3. Align on Common Metabolites")

        higher_mets = set(self.df_higher.columns) - {"DAS28", "Label"}
        lower_mets  = set(self.df_lower.columns)  - {"DAS28", "Label"}
        common_mets = sorted(higher_mets & lower_mets)

        print(f"  Metabolites in higher only : {len(higher_mets - lower_mets)}")
        print(f"  Metabolites in lower  only : {len(lower_mets - higher_mets)}")
        print(f"  Common metabolites         : {len(common_mets)}  <- using these")

        self.df_higher = self.df_higher[common_mets + ["DAS28", "Label"]]
        self.df_lower  = self.df_lower[common_mets  + ["DAS28", "Label"]]

    def combine_and_save(self) -> None:
        """Step 4 — Combine groups, drop DAS28, and save to CSV."""
        self.logger.section("4. Combine & Save")

        self.df_all = pd.concat(
            [self.df_higher, self.df_lower], axis=0
        ).reset_index(drop=True)
        self.df_all = self.df_all.drop(columns=["DAS28"])

        feature_cols = [c for c in self.df_all.columns if c != "Label"]
        n_high = (self.df_all["Label"] == 1).sum()
        n_low  = (self.df_all["Label"] == 0).sum()

        print(f"  Combined shape : {self.df_all.shape[0]} patients x {len(feature_cols)} metabolites + Label")
        print(f"  Label=1 (High) : {n_high} patients")
        print(f"  Label=0 (Low)  : {n_low}  patients")
        print(f"  Missing values : {self.df_all.isnull().sum().sum()}")

        if self.df_all.isnull().sum().sum() > 0:
            print("  WARNING: Missing values present — consider imputation before modelling.")

        self._plot_class_distribution()
        self.df_all.to_csv(self.processed_file, index=False)
        print(f"\n  Saved -> {self.processed_file}")

    def run(self) -> None:
        """Run all preprocessing steps in order."""
        self.load_and_parse()
        self.verify_group_separation()
        self.align_common_metabolites()
        self.combine_and_save()
        self.logger.section("Preprocessing Complete — run 03_model_a.py next")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    preprocessor = MetabolomicsPreprocessor(data_dir="data", output_dir="outputs")
    preprocessor.run()
