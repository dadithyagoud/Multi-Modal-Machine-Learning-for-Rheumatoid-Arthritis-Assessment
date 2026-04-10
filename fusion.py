"""
04_fusion.py — Late Fusion: Model A + Model B
==============================================
Class:
    FusionModel
        load_data()
        train_model_a()
        train_model_b()
        build_fusion_scores()
        fuse_weighted_average()
        fuse_stacking()
        plot_results()
        save_results()
        run()               ← runs all steps in order

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
    precision_score, recall_score, f1_score
)

from utils import Logger, FigureSaver


class FusionModel:
    """
    Implements late fusion of the metabolomics (Model A) and clinical
    (Model B) classifiers using two strategies:
        1. Weighted Average
        2. Stacking with a Logistic Regression meta-learner

    Parameters
    ----------
    data_dir     : directory containing both processed CSVs
    output_dir   : directory where plots and result CSV are saved
    random_state : seed for reproducibility
    test_size    : fraction of data held out for testing
    top_k        : top metabolites to select for Model A
    w_a          : weight for Model A in the weighted average (w_b = 1 - w_a)
    """

    def __init__(
        self,
        data_dir:     str   = "data",
        output_dir:   str   = "outputs",
        random_state: int   = 42,
        test_size:    float = 0.20,
        top_k:        int   = 50,
        w_a:          float = 0.35,
    ):
        self.data_dir     = data_dir
        self.output_dir   = output_dir
        self.random_state = random_state
        self.test_size    = test_size
        self.top_k        = top_k
        self.w_a          = w_a
        self.w_b          = round(1.0 - w_a, 10)

        assert abs(self.w_a + self.w_b - 1.0) < 1e-9, (
            f"Fusion weights must sum to 1.0 — got w_a={w_a} + w_b={self.w_b}"
        )

        self.logger = Logger()
        self.saver  = FigureSaver(output_dir)

        self.met_file    = os.path.join(data_dir, "metabolomics_processed.csv")
        self.clin_file   = os.path.join(data_dir, "clinical_processed.csv")
        self.results_file = os.path.join(output_dir, "fusion_results.csv")

        # Data
        self.df_met:       pd.DataFrame | None = None
        self.df_clin:      pd.DataFrame | None = None
        self.met_features: list[str]           = []
        self.clin_features: list[str]          = []
        self.X_met:  np.ndarray | None = None
        self.y_met:  np.ndarray | None = None
        self.X_clin: np.ndarray | None = None
        self.y_clin: np.ndarray | None = None

        # Model A
        self.rf_a:        RandomForestClassifier | None = None
        self.top_features: list[str]                    = []
        self.X_trA_sel:   np.ndarray | None             = None
        self.X_teA_sel:   np.ndarray | None             = None
        self.y_trA:       np.ndarray | None             = None
        self.y_teA:       np.ndarray | None             = None
        self.proba_a_all: np.ndarray | None             = None

        # Model B
        self.rf_b:         RandomForestClassifier | None = None
        self.X_trB:        np.ndarray | None             = None
        self.X_teB:        np.ndarray | None             = None
        self.y_trB:        np.ndarray | None             = None
        self.y_teB:        np.ndarray | None             = None
        self.proba_b_test: np.ndarray | None             = None
        self.proba_b_train: np.ndarray | None            = None

        # Fusion scores
        self.score_a_test:  np.ndarray | None = None
        self.score_a_train: np.ndarray | None = None
        self.score_b_test:  np.ndarray | None = None
        self.score_b_train: np.ndarray | None = None

        # Results
        self.fused_proba: np.ndarray | None = None
        self.fused_pred:  np.ndarray | None = None
        self.stack_proba: np.ndarray | None = None
        self.stack_pred:  np.ndarray | None = None
        self.meta_lr:     LogisticRegression | None = None
        self.m_a:   dict = {}
        self.m_b:   dict = {}
        self.m_fused: dict = {}
        self.m_stack: dict = {}

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    @staticmethod
    def _clf_metrics(y_true, y_pred, y_proba) -> dict:
        """Compute and return the standard 5-metric dict."""
        return {
            "Accuracy" : round(accuracy_score(y_true, y_pred),                   4),
            "ROC_AUC"  : round(roc_auc_score(y_true, y_proba),                   4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_true, y_pred,    zero_division=0), 4),
            "F1"       : round(f1_score(y_true, y_pred,        zero_division=0), 4),
        }

    # ------------------------------------------------------------------
    # Public pipeline steps
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Step 1 — Load both processed datasets."""
        self.logger.section("1. Load Both Processed Datasets")

        for fpath in [self.met_file, self.clin_file]:
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Input file not found: {fpath}\n"
                    f"Run both preprocessing scripts first."
                )

        self.df_met  = pd.read_csv(self.met_file)
        self.df_clin = pd.read_csv(self.clin_file)

        self.met_features  = [c for c in self.df_met.columns  if c != "Label"]
        self.clin_features = [c for c in self.df_clin.columns if c != "Label"]

        self.X_met  = self.df_met[self.met_features].values
        self.y_met  = self.df_met["Label"].values
        self.X_clin = self.df_clin[self.clin_features].values
        self.y_clin = self.df_clin["Label"].values

        print(f"  Model A data : {self.df_met.shape[0]} patients  | {len(self.met_features)} metabolites")
        print(f"    Label=0 (Low DAS28)  : {(self.y_met==0).sum()}")
        print(f"    Label=1 (High DAS28) : {(self.y_met==1).sum()}")
        print(f"\n  Model B data : {self.df_clin.shape[0]} patients  | {len(self.clin_features)} features")
        print(f"    Label=0 (Normal) : {(self.y_clin==0).sum()}")
        print(f"    Label=1 (RA)     : {(self.y_clin==1).sum()}")

    def train_model_a(self) -> None:
        """Step 2 — Retrain Model A (metabolomics RF) with feature selection."""
        self.logger.section("2. Retrain Model A — RF Classifier (Metabolomics -> High/Low DAS28)")

        X_trA, X_teA, self.y_trA, self.y_teA = train_test_split(
            self.X_met, self.y_met,
            test_size=self.test_size, random_state=self.random_state, stratify=self.y_met
        )

        # Feature selection on training set only
        rf_a_full = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )
        rf_a_full.fit(X_trA, self.y_trA)

        imp_a             = pd.Series(rf_a_full.feature_importances_, index=self.met_features)
        self.top_features = imp_a.nlargest(self.top_k).index.tolist()

        # O(n) index map
        feat_idx_map  = {f: i for i, f in enumerate(self.met_features)}
        top_idx       = [feat_idx_map[f] for f in self.top_features]

        self.X_trA_sel = X_trA[:, top_idx]
        self.X_teA_sel = X_teA[:, top_idx]
        X_met_sel      = self.X_met[:, top_idx]

        self.rf_a = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )
        self.rf_a.fit(self.X_trA_sel, self.y_trA)

        proba_a_test     = self.rf_a.predict_proba(self.X_teA_sel)[:, 1]
        self.proba_a_all = self.rf_a.predict_proba(X_met_sel)[:, 1]

        self.m_a = self._clf_metrics(self.y_teA, self.rf_a.predict(self.X_teA_sel), proba_a_test)
        print(f"  Model A — Accuracy={self.m_a['Accuracy']:.4f}  "
              f"AUC={self.m_a['ROC_AUC']:.4f}  F1={self.m_a['F1']:.4f}")
        print(f"  P(High DAS28) — mean: {self.proba_a_all.mean():.4f}  "
              f"std: {self.proba_a_all.std():.4f}")

    def train_model_b(self) -> None:
        """Step 3 — Retrain Model B (clinical RF)."""
        self.logger.section("3. Retrain Model B — RF Classifier (Clinical -> RA)")

        self.X_trB, self.X_teB, self.y_trB, self.y_teB = train_test_split(
            self.X_clin, self.y_clin,
            test_size=self.test_size, random_state=self.random_state, stratify=self.y_clin
        )

        self.rf_b = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )
        self.rf_b.fit(self.X_trB, self.y_trB)

        self.proba_b_test  = self.rf_b.predict_proba(self.X_teB)[:, 1]
        self.proba_b_train = self.rf_b.predict_proba(self.X_trB)[:, 1]

        self.m_b = self._clf_metrics(self.y_teB, self.rf_b.predict(self.X_teB), self.proba_b_test)
        print(f"  Model B — Accuracy={self.m_b['Accuracy']:.4f}  "
              f"AUC={self.m_b['ROC_AUC']:.4f}  F1={self.m_b['F1']:.4f}")

    def build_fusion_scores(self) -> None:
        """Step 4 — Sample Model A scores for clinical cohort patients."""
        self.logger.section("4. Build Fusion Scores")

        rng = np.random.default_rng(self.random_state)
        self.score_a_test  = rng.choice(self.proba_a_all, size=len(self.y_teB), replace=True)
        self.score_a_train = rng.choice(self.proba_a_all, size=len(self.y_trB), replace=True)
        self.score_b_test  = self.proba_b_test
        self.score_b_train = self.proba_b_train

        print(f"  Score A [P(High DAS28)] — test mean: {self.score_a_test.mean():.4f}  "
              f"std: {self.score_a_test.std():.4f}")
        print(f"  Score B [P(RA)]         — test mean: {self.score_b_test.mean():.4f}  "
              f"std: {self.score_b_test.std():.4f}")

    def fuse_weighted_average(self) -> None:
        """Step 5 — Weighted average fusion."""
        self.logger.section(f"5. Fusion Strategy 1 — Weighted Average  (w_A={self.w_a}, w_B={self.w_b})")

        self.fused_proba = self.w_a * self.score_a_test + self.w_b * self.score_b_test
        self.fused_pred  = (self.fused_proba >= 0.5).astype(int)
        self.m_fused     = self._clf_metrics(self.y_teB, self.fused_pred, self.fused_proba)

        print(f"\n  Accuracy  : {self.m_fused['Accuracy']:.4f}")
        print(f"  ROC-AUC   : {self.m_fused['ROC_AUC']:.4f}")
        print(f"  Precision : {self.m_fused['Precision']:.4f}")
        print(f"  Recall    : {self.m_fused['Recall']:.4f}")
        print(f"  F1 Score  : {self.m_fused['F1']:.4f}")

    def fuse_stacking(self) -> None:
        """Step 6 — Stacking with Logistic Regression meta-learner."""
        self.logger.section("6. Fusion Strategy 2 — Stacking (Logistic Regression meta-learner)")

        meta_train = np.column_stack([self.score_a_train, self.score_b_train])
        meta_test  = np.column_stack([self.score_a_test,  self.score_b_test])

        self.meta_lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.meta_lr.fit(meta_train, self.y_trB)

        self.stack_pred  = self.meta_lr.predict(meta_test)
        self.stack_proba = self.meta_lr.predict_proba(meta_test)[:, 1]
        self.m_stack     = self._clf_metrics(self.y_teB, self.stack_pred, self.stack_proba)

        print(f"\n  Meta-learner coefficients:")
        print(f"    Score A [metabolomics] : {self.meta_lr.coef_[0][0]:+.4f}")
        print(f"    Score B [clinical]     : {self.meta_lr.coef_[0][1]:+.4f}")
        print(f"    Intercept              : {self.meta_lr.intercept_[0]:+.4f}")
        print(f"\n  Accuracy  : {self.m_stack['Accuracy']:.4f}")
        print(f"  ROC-AUC   : {self.m_stack['ROC_AUC']:.4f}")
        print(f"  Precision : {self.m_stack['Precision']:.4f}")
        print(f"  Recall    : {self.m_stack['Recall']:.4f}")
        print(f"  F1 Score  : {self.m_stack['F1']:.4f}")

    def plot_results(self) -> None:
        """Step 7 — Generate all five output plots."""
        self.logger.section("7. Generating Plots")

        self._plot_roc_comparison()
        self._plot_metrics_comparison()
        self._plot_confusion_matrices()
        self._plot_score_distributions()
        self._plot_metalearner_weights()

    def _plot_roc_comparison(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        for label, y_score, color, ls in [
            (f"Model B only  (AUC={self.m_b['ROC_AUC']:.3f})",           self.proba_b_test, "steelblue",  "-"),
            (f"Fusion Weighted Avg  (AUC={self.m_fused['ROC_AUC']:.3f})", self.fused_proba,  "darkorange", "--"),
            (f"Fusion Stacking  (AUC={self.m_stack['ROC_AUC']:.3f})",     self.stack_proba,  "seagreen",   "-."),
        ]:
            fpr, tpr, _ = roc_curve(self.y_teB, y_score)
            ax.plot(fpr, tpr, lw=2, label=label, color=color, linestyle=ls)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Fusion — ROC Curve Comparison")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        self.saver.save("F_01_roc_comparison.png")

    def _plot_metrics_comparison(self) -> None:
        metrics     = ["Accuracy", "ROC_AUC", "Precision", "Recall", "F1"]
        labels_met  = ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"]
        vals_b      = [self.m_b[k]     for k in metrics]
        vals_fused  = [self.m_fused[k] for k in metrics]
        vals_stack  = [self.m_stack[k] for k in metrics]

        x = np.arange(len(metrics))
        w = 0.25
        fig, ax = plt.subplots(figsize=(11, 5))
        b1 = ax.bar(x - w, vals_b,     w, label="Model B only",       color="steelblue",  edgecolor="white")
        b2 = ax.bar(x,     vals_fused, w, label="Fusion Weighted Avg", color="darkorange", edgecolor="white")
        b3 = ax.bar(x + w, vals_stack, w, label="Fusion Stacking",     color="seagreen",   edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_met)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title("Fusion — Full Metrics Comparison")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        for bars in [b1, b2, b3]:
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        self.saver.save("F_02_metrics_comparison.png")

    def _plot_confusion_matrices(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, y_pred, title in zip(
            axes,
            [self.rf_b.predict(self.X_teB), self.fused_pred, self.stack_pred],
            [
                "Model B only",
                f"Fusion Weighted Avg\n(w_A={self.w_a}, w_B={self.w_b})",
                "Fusion Stacking\n(LR meta-learner)"
            ]
        ):
            ConfusionMatrixDisplay(
                confusion_matrix(self.y_teB, y_pred),
                display_labels=["Normal", "RA"]
            ).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(title)
        plt.suptitle("Fusion — Confusion Matrix Comparison",
                     fontsize=13, fontweight="bold", y=1.02)
        self.saver.save("F_03_confusion_matrices.png")

    def _plot_score_distributions(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, scores, title, ra_color in zip(
            axes,
            [self.score_a_test, self.score_b_test],
            ["Model A — P(High DAS28)\n(metabolomics signal)",
             "Model B — P(RA)\n(clinical signal)"],
            ["tomato", "steelblue"]
        ):
            for lbl, mask, alpha in [
                (f"Normal (n={(self.y_teB==0).sum()})", self.y_teB == 0, 0.5),
                (f"RA     (n={(self.y_teB==1).sum()})", self.y_teB == 1, 0.6)
            ]:
                bar_color = ra_color if "RA" in lbl else "slategray"
                ax.hist(scores[mask], bins=20, alpha=alpha, label=lbl,
                        density=True, color=bar_color)
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        plt.suptitle("Fusion — Score Distributions by Class",
                     fontsize=13, fontweight="bold", y=1.02)
        self.saver.save("F_04_score_distributions.png")

    def _plot_metalearner_weights(self) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        coefs = [abs(self.meta_lr.coef_[0][0]), abs(self.meta_lr.coef_[0][1])]
        ax.bar(["Model A\n(Metabolomics)", "Model B\n(Clinical)"],
               coefs, color=["tomato", "steelblue"], edgecolor="white", width=0.4)
        ax.set_ylabel("Absolute Coefficient")
        ax.set_title("Stacking — Meta-learner Weights\n(how much each modality contributes)")
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(coefs):
            ax.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
        self.saver.save("F_05_metalearner_weights.png")

    def save_results(self) -> None:
        """Step 8 — Save all strategy metrics to CSV."""
        self.logger.section("8. Save Results Summary")

        results_df = pd.DataFrame([
            {"Strategy": "Model A only (Metabolomics)",             **self.m_a,     "Notes": "RF on metabolomics, High vs Low DAS28"},
            {"Strategy": "Model B only (Clinical)",                 **self.m_b,     "Notes": "RF on clinical features, RA vs Normal"},
            {"Strategy": f"Fusion Weighted Avg (w_A={self.w_a}, w_B={self.w_b})", **self.m_fused, "Notes": "Weighted average of P(High DAS28) + P(RA)"},
            {"Strategy": "Fusion Stacking (LR meta-learner)",       **self.m_stack, "Notes": "LR trained on [score_A, score_B] -> RA label"},
        ])
        results_df.to_csv(self.results_file, index=False)

        print(f"\n  Saved -> {self.results_file}")
        print(f"\n  {'Strategy':<42} {'Acc':>6} {'AUC':>6} {'F1':>6}")
        print(f"  {'-'*58}")
        for _, row in results_df.iterrows():
            print(f"  {row['Strategy']:<42} {row['Accuracy']:>6.4f} {row['ROC_AUC']:>6.4f} {row['F1']:>6.4f}")

        print(f"\n  Meta-learner weights:")
        print(f"    Metabolomics coeff : {self.meta_lr.coef_[0][0]:+.4f}")
        print(f"    Clinical coeff     : {self.meta_lr.coef_[0][1]:+.4f}")

    def run(self) -> None:
        """Run all fusion steps in order."""
        self.load_data()
        self.train_model_a()
        self.train_model_b()
        self.build_fusion_scores()
        self.fuse_weighted_average()
        self.fuse_stacking()
        self.plot_results()
        self.save_results()
        self.logger.section("Fusion Complete")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    fusion = FusionModel(data_dir="data", output_dir="outputs")
    fusion.run()
