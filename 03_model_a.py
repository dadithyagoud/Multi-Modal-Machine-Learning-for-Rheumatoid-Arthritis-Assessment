"""
03_model_a.py — Model A: Random Forest Classifier (Metabolomics -> High/Low DAS28)
====================================================================================
Dataset : data/metabolomics_processed.csv  (output of 02_preprocessing_a.py)

Class:
    MetabolomicsClassifier
        load_data()
        split_data()
        select_features()
        train()
        evaluate()
        cross_validate()
        plot_feature_importance()
        save_results()
        run()              ← runs all steps in order

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

from utils import Logger, FigureSaver


class MetabolomicsClassifier:
    """
    Trains and evaluates a Random Forest classifier on metabolomics data
    to distinguish High DAS28 (label=1) from Low DAS28 (label=0).

    Parameters
    ----------
    data_dir     : directory containing metabolomics_processed.csv
    output_dir   : directory where plots and result CSV are saved
    random_state : seed for reproducibility
    test_size    : fraction of data held out for testing
    top_k        : number of metabolites selected by feature importance
    """

    def __init__(
        self,
        data_dir:     str = "data",
        output_dir:   str = "outputs",
        random_state: int = 42,
        test_size:    float = 0.20,
        top_k:        int = 50,
    ):
        self.data_dir     = data_dir
        self.output_dir   = output_dir
        self.random_state = random_state
        self.test_size    = test_size
        self.top_k        = top_k

        self.logger = Logger()
        self.saver  = FigureSaver(output_dir)

        self.processed_file = os.path.join(data_dir, "metabolomics_processed.csv")
        self.results_file   = os.path.join(output_dir, "A_model_results.csv")

        # Populated during run
        self.df:          pd.DataFrame | None = None
        self.feature_cols: list[str]          = []
        self.X:  np.ndarray | None            = None
        self.y:  np.ndarray | None            = None

        self.X_train:     np.ndarray | None = None
        self.X_test:      np.ndarray | None = None
        self.y_train:     np.ndarray | None = None
        self.y_test:      np.ndarray | None = None

        self.top_features: list[str]        = []
        self.top_idx:      list[int]        = []
        self.X_train_sel:  np.ndarray | None = None
        self.X_test_sel:   np.ndarray | None = None
        self.X_sel:        np.ndarray | None = None

        self.model: RandomForestClassifier | None = None

        # Evaluation results
        self.y_pred:  np.ndarray | None = None
        self.y_proba: np.ndarray | None = None
        self.metrics: dict = {}

        # CV results
        self.cv_acc: np.ndarray | None = None
        self.cv_auc: np.ndarray | None = None
        self.cv_f1:  np.ndarray | None = None

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public pipeline steps
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Step 1 — Load the processed metabolomics CSV."""
        self.logger.section("1. Load Processed Data")

        if not os.path.exists(self.processed_file):
            raise FileNotFoundError(
                f"Input file not found: {self.processed_file}\n"
                f"Run 02_preprocessing_a.py first."
            )

        self.df           = pd.read_csv(self.processed_file)
        self.feature_cols = [c for c in self.df.columns if c != "Label"]
        self.X            = self.df[self.feature_cols].values
        self.y            = self.df["Label"].values

        print(f"  Patients    : {self.df.shape[0]}")
        print(f"  Metabolites : {len(self.feature_cols)}")
        print(f"  Label=0 (Low DAS28)  : {(self.y==0).sum()}")
        print(f"  Label=1 (High DAS28) : {(self.y==1).sum()}")

    def split_data(self) -> None:
        """Step 2 — Stratified 80/20 train/test split."""
        self.logger.section("2. Train / Test Split  (80/20, stratified)")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )

        print(f"  Train : {self.X_train.shape[0]} patients  "
              f"| Low={(self.y_train==0).sum()}  High={(self.y_train==1).sum()}")
        print(f"  Test  : {self.X_test.shape[0]}  patients  "
              f"| Low={(self.y_test==0).sum()}   High={(self.y_test==1).sum()}")

    def select_features(self) -> None:
        """Step 3 — Select top-K metabolites by RF importance on training data."""
        self.logger.section(f"3. Feature Selection  (top {self.top_k} metabolites)")

        rf_full = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced",
            random_state=self.random_state, n_jobs=-1
        )
        rf_full.fit(self.X_train, self.y_train)

        importances       = pd.Series(rf_full.feature_importances_, index=self.feature_cols)
        self.top_features = importances.nlargest(self.top_k).index.tolist()

        # O(n) index lookup map instead of repeated linear scans
        feat_idx_map  = {f: i for i, f in enumerate(self.feature_cols)}
        self.top_idx  = [feat_idx_map[f] for f in self.top_features]

        self.X_train_sel = self.X_train[:, self.top_idx]
        self.X_test_sel  = self.X_test[:,  self.top_idx]
        self.X_sel       = self.X[:,       self.top_idx]

        print(f"  Selected {self.top_k} metabolites from {len(self.feature_cols)}")
        print(f"  Top 10 : {self.top_features[:10]}")

    def train(self) -> None:
        """Step 4 — Train final RF classifier on selected features."""
        self.logger.section(f"4. Train Final RF Classifier  (top {self.top_k} metabolites)")

        self.model = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced",
            random_state=self.random_state, n_jobs=-1
        )
        self.model.fit(self.X_train_sel, self.y_train)
        print(f"  Trained: {self.model.n_estimators} trees  |  class_weight='balanced'")

    def evaluate(self) -> None:
        """Step 5 — Evaluate on test set; save confusion matrix and ROC curve."""
        self.logger.section("5. Test Set Evaluation")

        self.y_pred  = self.model.predict(self.X_test_sel)
        self.y_proba = self.model.predict_proba(self.X_test_sel)[:, 1]

        self.metrics = {
            "Accuracy" : accuracy_score(self.y_test, self.y_pred),
            "ROC_AUC"  : roc_auc_score(self.y_test, self.y_proba),
            "Precision": precision_score(self.y_test, self.y_pred, zero_division=0),
            "Recall"   : recall_score(self.y_test, self.y_pred,    zero_division=0),
            "F1"       : f1_score(self.y_test, self.y_pred,        zero_division=0),
        }

        print(f"\n  Accuracy  : {self.metrics['Accuracy']:.4f}")
        print(f"  ROC-AUC   : {self.metrics['ROC_AUC']:.4f}")
        print(f"  Precision : {self.metrics['Precision']:.4f}")
        print(f"  Recall    : {self.metrics['Recall']:.4f}")
        print(f"  F1 Score  : {self.metrics['F1']:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(
            self.y_test, self.y_pred,
            target_names=["Low DAS28", "High DAS28"], zero_division=0
        ))

        self._plot_confusion_matrix()
        self._plot_roc_curve()

    def _plot_confusion_matrix(self) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(self.y_test, self.y_pred),
            display_labels=["Low DAS28", "High DAS28"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Model A — Confusion Matrix\nAccuracy={self.metrics['Accuracy']:.3f}")
        self.saver.save("A_03_confusion_matrix.png")

    def _plot_roc_curve(self) -> None:
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="tomato", lw=2,
                label=f"RF Classifier  (AUC = {self.metrics['ROC_AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
        ax.fill_between(fpr, tpr, alpha=0.08, color="tomato")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Model A — ROC Curve (High vs Low DAS28)")
        ax.legend(loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        self.saver.save("A_04_roc_curve.png")

    def cross_validate(self) -> None:
        """Step 6 — Stratified 5-fold cross-validation; save bar chart."""
        self.logger.section("6. Cross-Validation  (Stratified 5-Fold)")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # NOTE: Feature selection was done before CV — minor optimistic bias
        # acknowledged as a known limitation. A Pipeline would fully prevent it.
        self.cv_acc = cross_val_score(self.model, self.X_sel, self.y, cv=cv, scoring="accuracy", n_jobs=-1)
        self.cv_auc = cross_val_score(self.model, self.X_sel, self.y, cv=cv, scoring="roc_auc",  n_jobs=-1)
        self.cv_f1  = cross_val_score(self.model, self.X_sel, self.y, cv=cv, scoring="f1",       n_jobs=-1)

        print(f"\n  {'Metric':<12} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8} {'Mean':>8} {'Std':>8}")
        print(f"  {'-'*70}")
        for name, scores in [("Accuracy", self.cv_acc), ("ROC-AUC", self.cv_auc), ("F1", self.cv_f1)]:
            vals = "  ".join([f"{s:.4f}" for s in scores])
            print(f"  {name:<12} {vals}   {scores.mean():.4f}   {scores.std():.4f}")

        self._plot_cv_results()

    def _plot_cv_results(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(5)
        w = 0.25
        ax.bar(x - w, self.cv_acc, w, label="Accuracy", color="tomato",     edgecolor="white")
        ax.bar(x,     self.cv_auc, w, label="ROC-AUC",  color="darkorange", edgecolor="white")
        ax.bar(x + w, self.cv_f1,  w, label="F1",       color="steelblue",  edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title("Model A — 5-Fold Cross-Validation Results")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        for bars in ax.containers:
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        self.saver.save("A_05_cross_validation.png")

    def plot_feature_importance(self) -> None:
        """Step 7 — Plot top-20 metabolite importances from the final model."""
        self.logger.section("7. Feature Importance  (top 20 metabolites)")

        imp     = pd.Series(self.model.feature_importances_, index=self.top_features).sort_values()
        top20   = imp.nlargest(20).sort_values()
        colors  = ["tomato" if v == top20.max() else "steelblue" for v in top20.values]

        fig, ax = plt.subplots(figsize=(8, 6))
        top20.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Model A — Top 20 Metabolite Importances\n(red = most important)")
        ax.set_xlabel("Importance")
        ax.spines[["top", "right"]].set_visible(False)
        self.saver.save("A_06_feature_importance.png")

        print(f"\n  Top 10 metabolites:")
        print(imp.nlargest(10).round(5).to_string())

    def save_results(self) -> None:
        """Step 8 — Save all metrics to CSV."""
        self.logger.section("8. Save Results")

        results_df = pd.DataFrame([{
            "Model"          : "Model A — RF Classifier",
            "Dataset"        : "Hur et al. 2021 (High vs Low DAS28)",
            "Task"           : "High DAS28 (1) vs Low DAS28 (0)",
            "n_total"        : len(self.df),
            "n_train"        : self.X_train.shape[0],
            "n_test"         : self.X_test.shape[0],
            "n_features_raw" : len(self.feature_cols),
            "n_features_sel" : self.top_k,
            "Accuracy"       : round(self.metrics["Accuracy"],  4),
            "ROC_AUC"        : round(self.metrics["ROC_AUC"],   4),
            "Precision"      : round(self.metrics["Precision"], 4),
            "Recall"         : round(self.metrics["Recall"],    4),
            "F1"             : round(self.metrics["F1"],        4),
            "CV_Acc_mean"    : round(self.cv_acc.mean(), 4),
            "CV_Acc_std"     : round(self.cv_acc.std(),  4),
            "CV_AUC_mean"    : round(self.cv_auc.mean(), 4),
            "CV_AUC_std"     : round(self.cv_auc.std(),  4),
            "CV_F1_mean"     : round(self.cv_f1.mean(),  4),
            "CV_F1_std"      : round(self.cv_f1.std(),   4),
        }])
        results_df.to_csv(self.results_file, index=False)

        print(f"\n  Saved -> {self.results_file}")
        print(f"\n  Summary:")
        print(f"    Accuracy : {self.metrics['Accuracy']:.4f}   "
              f"(CV: {self.cv_acc.mean():.4f} +/- {self.cv_acc.std():.4f})")
        print(f"    ROC-AUC  : {self.metrics['ROC_AUC']:.4f}   "
              f"(CV: {self.cv_auc.mean():.4f} +/- {self.cv_auc.std():.4f})")
        print(f"    F1       : {self.metrics['F1']:.4f}   "
              f"(CV: {self.cv_f1.mean():.4f} +/- {self.cv_f1.std():.4f})")

    def run(self) -> None:
        """Run all training and evaluation steps in order."""
        self.load_data()
        self.split_data()
        self.select_features()
        self.train()
        self.evaluate()
        self.cross_validate()
        self.plot_feature_importance()
        self.save_results()
        self.logger.section("Model A Complete")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    classifier = MetabolomicsClassifier(data_dir="data", output_dir="outputs")
    classifier.run()
