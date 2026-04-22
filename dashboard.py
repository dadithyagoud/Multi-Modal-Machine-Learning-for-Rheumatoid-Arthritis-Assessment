"""
05_dashboard.py — FusionRA Dashboard
=====================================
Run with: streamlit run 05_dashboard.py

Classes:
    DashboardConfig   — holds all constants and style strings
    ModelCache        — wraps @st.cache_resource model training
    HomePage          — renders the Home page
    MetaRAPage        — renders the MetaRA page
    ClinicalRAPage    — renders the ClinicalRA page
    FusionRAPage      — renders the FusionRA page
    Dashboard         — top-level controller: sidebar + page routing
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


# =============================================================================
# DashboardConfig
# =============================================================================
class DashboardConfig:
    """
    Holds all static configuration: file paths, model hyper-parameters,
    fusion presets, and CSS style strings.
    """

    DATA_DIR     = "data"
    RANDOM_STATE = 42
    TEST_SIZE    = 0.20
    TOP_K        = 50

    PRESETS = {
        "Clinical-Heavy (30/70)":       0.30,
        "Balanced (50/50)":             0.50,
        "Metabolomics-Heavy (70/30)":   0.70,
        "Clinical Only (0/100)":        0.00,
    }

    CSS = """
<style>
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] {
        transform: none !important;
        width: 18rem !important;
        min-width: 18rem !important;
    }
    section[data-testid="stSidebar"] > div { width: 18rem !important; }
    .main .block-container { margin-left: 1rem !important; padding-left: 2rem !important; }
    .stApp { background-color: #F5F7FA; }
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0D1B2A 0%, #1B3A5C 100%) !important;
        border-right: 2px solid #1B998B !important;
    }
    [data-testid="stSidebar"] .stRadio label p {
        color: #FFFFFF !important; font-size: 15px !important; font-weight: 500 !important;
    }
    [data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E2E8F0; border-top:3px solid #1B998B; border-radius:8px; padding:12px 16px; }
    [data-testid="stMetricLabel"] p { color:#6B7280 !important; font-size:12px !important; font-weight:600 !important; }
    [data-testid="stMetricValue"]   { color:#0D1B2A !important; font-weight:700 !important; }
    [data-testid="stMetricDelta"]   { color:#1B998B !important; }
    .sec { background:#EBF4FF; color:#0D1B2A !important; padding:10px 18px; border-radius:6px; border-left:4px solid #1B998B; font-size:15px; font-weight:700; margin:18px 0 12px 0; }
    .card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:18px 20px; margin-bottom:12px; }
    .card h4 { color:#0D1B2A !important; font-size:16px; margin:0 0 10px 0; }
    .card p  { color:#374151 !important; font-size:14px; line-height:1.7; }
    .card b  { color:#0D1B2A !important; }
    .mode-card    { background:#FFFFFF; border:2px solid #E2E8F0; border-radius:12px; padding:24px 20px; text-align:center; }
    .mode-card-a  { border-top:5px solid #1B998B; }
    .mode-card-b  { border-top:5px solid #F4A261; }
    .mode-card-ab { border-top:5px solid #E84855; }
    .mode-title   { font-size:20px; font-weight:700; color:#0D1B2A !important; margin-bottom:8px; }
    .mode-sub     { font-size:13px; color:#6B7280 !important; line-height:1.6; }
    .mode-req-a   { color:#1B998B !important; font-weight:600; }
    .mode-req-b   { color:#E07A30 !important; font-weight:600; }
    .mode-req-ab  { color:#E84855 !important; font-weight:600; }
    .pred-ra     { background:#FFF0F2; border:2px solid #E84855; border-radius:10px; padding:20px; text-align:center; }
    .pred-normal { background:#F0FFF8; border:2px solid #1B998B; border-radius:10px; padding:20px; text-align:center; }
    .pred-ra .pred-title     { color:#C0152A !important; font-size:22px; font-weight:700; }
    .pred-ra .pred-score     { color:#C0152A !important; font-size:15px; }
    .pred-normal .pred-title { color:#0D6B5E !important; font-size:22px; font-weight:700; }
    .pred-normal .pred-score { color:#0D6B5E !important; font-size:15px; }
    .warn      { background:#FFFBEB; border-left:4px solid #F4A261; border-radius:8px; padding:14px 18px; margin-top:12px; color:#7C4A00 !important; font-size:14px; }
    .info-note { background:#EFF8FF; border-left:4px solid #4A90D9; border-radius:8px; padding:14px 18px; margin-top:4px; color:#1A3A6A !important; font-size:14px; }
    .stNumberInput label, .stSelectbox label, .stSlider label,
    .stFileUploader label, .stRadio label, .stTextInput label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label {
        color: #0D1B2A !important; font-size: 14px !important; font-weight: 500 !important;
    }
    .stNumberInput input { background-color: #FFFFFF !important; color: #0D1B2A !important; border: 1px solid #CBD5E0 !important; border-radius: 6px !important; }
    .stNumberInput > div, .stNumberInput > div > div { background-color: #FFFFFF !important; border: 1px solid #CBD5E0 !important; border-radius: 6px !important; }
    .stNumberInput button { background-color: #E2E8F0 !important; color: #0D1B2A !important; border: none !important; }
    .stNumberInput button:hover { background-color: #CBD5E0 !important; }
    .stSelectbox > div > div { background-color: #FFFFFF !important; color: #0D1B2A !important; border: 1px solid #CBD5E0 !important; border-radius: 6px !important; }
    .stSelectbox div[data-baseweb="select"] span, .stSelectbox div[data-baseweb="select"] div { color: #0D1B2A !important; background-color: #FFFFFF !important; }
    .stSelectbox svg { fill: #0D1B2A !important; }
    [data-baseweb="select"] svg { fill: #0D1B2A !important; }
    .stSlider > div > div > div { background-color: #E2E8F0 !important; }
    .main p, .main li { color: #374151 !important; }
    .main strong, .main b { color: #0D1B2A !important; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
"""


# =============================================================================
# ModelCache
# =============================================================================
class ModelCache:
    """
    Wraps Streamlit's caching decorators.

    load_data()     — cached data loader (returns df_met, df_clin)
    train_models()  — cached model trainer (returns all trained artefacts)
    """

    cfg = DashboardConfig

    @staticmethod
    @st.cache_data
    def load_data():
        """Load both processed CSVs; stop the app with an error if missing."""
        met_path  = os.path.join(DashboardConfig.DATA_DIR, "metabolomics_processed.csv")
        clin_path = os.path.join(DashboardConfig.DATA_DIR, "clinical_processed.csv")
        missing   = [p for p in [met_path, clin_path] if not os.path.exists(p)]
        if missing:
            st.error(
                f"**Missing data files:** {missing}\n\n"
                f"Run the preprocessing scripts first:\n"
                f"```\npython 02_preprocessing_a.py\npython 02_preprocessing_b.py\n```"
            )
            st.stop()
        return pd.read_csv(met_path), pd.read_csv(clin_path)

    @staticmethod
    @st.cache_resource
    def train_models():
        """Train Model A (with feature selection) and Model B; return all artefacts."""
        cfg = DashboardConfig
        df_met, df_clin = ModelCache.load_data()

        # --- Model A ---
        mf        = [c for c in df_met.columns  if c != "Label"]
        Xm, ym    = df_met[mf].values, df_met["Label"].values
        XtrA, XteA, ytrA, yteA = train_test_split(
            Xm, ym, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=ym
        )
        rf_full = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=cfg.RANDOM_STATE, n_jobs=-1
        )
        rf_full.fit(XtrA, ytrA)
        imp      = pd.Series(rf_full.feature_importances_, index=mf)
        top_f    = imp.nlargest(cfg.TOP_K).index.tolist()
        feat_idx = {f: i for i, f in enumerate(mf)}
        tidx     = [feat_idx[f] for f in top_f]
        XtrAs, XteAs, Xms = XtrA[:, tidx], XteA[:, tidx], Xm[:, tidx]

        rf_a = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=cfg.RANDOM_STATE, n_jobs=-1
        )
        rf_a.fit(XtrAs, ytrA)

        # --- Model B ---
        cf     = [c for c in df_clin.columns if c != "Label"]
        Xc, yc = df_clin[cf].values, df_clin["Label"].values
        XtrB, XteB, ytrB, yteB = train_test_split(
            Xc, yc, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=yc
        )
        rf_b = RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=cfg.RANDOM_STATE, n_jobs=-1
        )
        rf_b.fit(XtrB, ytrB)

        return (
            rf_a, top_f, tidx, mf, XteAs, yteA,
            rf_b, cf, XteB, yteB,
            rf_a.predict_proba(Xms)[:, 1]
        )


# =============================================================================
# Page classes
# =============================================================================
class HomePage:
    """Renders the Home page with dataset info and results summary."""

    def render(self, models: tuple) -> None:
        st.markdown("<h1 style='color:#0D1B2A'>FusionRA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B7280; font-size:16px'>Multi-Modal Machine Learning for Rheumatoid Arthritis Analysis</p>", unsafe_allow_html=True)
        st.markdown("---")

        self._render_mode_cards()
        self._render_datasets()
        self._render_results_summary()

    def _render_mode_cards(self) -> None:
        st.markdown("<div class='sec'>Select a Tool</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B7280'>Use the sidebar on the left to navigate between tools.</p>", unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("""<div class='mode-card mode-card-a'>
                <div style='font-size:36px; margin-bottom:10px'>🔬</div>
                <div class='mode-title'>MetaRA</div>
                <div class='mode-sub'>Predict High vs Low DAS28 from metabolomics data.<br><br>
                <span class='mode-req-a'>Requires:</span> Metabolomics file (.csv/.tsv)</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open MetaRA", key="home_a", use_container_width=True):
                st.session_state["nav"] = "MetaRA"; st.rerun()
        with mc2:
            st.markdown("""<div class='mode-card mode-card-b'>
                <div style='font-size:36px; margin-bottom:10px'>🏥</div>
                <div class='mode-title'>ClinicalRA</div>
                <div class='mode-sub'>Classify RA vs Normal using clinical biomarkers.<br><br>
                <span class='mode-req-b'>Requires:</span> Clinical values (ESR, CRP, RF etc.)</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open ClinicalRA", key="home_b", use_container_width=True):
                st.session_state["nav"] = "ClinicalRA"; st.rerun()
        with mc3:
            st.markdown("""<div class='mode-card mode-card-ab'>
                <div style='font-size:36px; margin-bottom:10px'>🔗</div>
                <div class='mode-title'>FusionRA</div>
                <div class='mode-sub'>Combine both models with adjustable weights.<br><br>
                <span class='mode-req-ab'>Requires:</span> Both metabolomics + clinical data</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open FusionRA", key="home_ab", use_container_width=True):
                st.session_state["nav"] = "FusionRA"; st.rerun()

    def _render_datasets(self) -> None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sec'>Datasets</div>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("""<div class='card'><h4>Dataset A — Metabolomics</h4>
            <p><b>Source:</b> Hur et al. (2021), Arthritis Research & Therapy</p>
            <p><b>Patients:</b> 128 (52 high DAS28 · 76 low DAS28)</p>
            <p><b>Features:</b> 686 metabolites → top 50 selected</p>
            <p><b>DOI:</b> 10.1186/s13075-021-02537-4</p></div>""", unsafe_allow_html=True)
        with d2:
            st.markdown("""<div class='card'><h4>Dataset B — Clinical</h4>
            <p><b>Source:</b> Mahdi et al. (2025), Harvard Dataverse</p>
            <p><b>Patients:</b> 4,452 (2,848 RA · 1,604 Normal)</p>
            <p><b>Features:</b> 14 (ESR, CRP, RF, Anti-CCP, ANA, HLA-B27 etc.)</p>
            <p><b>DOI:</b> 10.7910/DVN/VM4OR3</p></div>""", unsafe_allow_html=True)

    def _render_results_summary(self) -> None:
        st.markdown("<div class='sec'>Results Summary</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Tool"         : ["MetaRA", "ClinicalRA", "FusionRA — Weighted Avg", "FusionRA — Stacking"],
            "Task"         : ["High vs Low DAS28", "RA vs Normal", "RA vs Normal", "RA vs Normal"],
            "CV AUC"       : ["0.839 ± 0.079", "1.000 ± 0.000", "—", "—"],
            "Test Accuracy": ["0.654", "1.000", "1.000", "1.000"],
            "Test AUC"     : ["0.676", "1.000", "1.000", "1.000"],
        }), use_container_width=True, hide_index=True)
        st.markdown("""<div class='warn'><b>Note:</b> ClinicalRA perfect scores reflect near-perfect class
        separation in the Mahdi et al. dataset — an acknowledged limitation. MetaRA CV AUC 0.839
        reflects a genuine metabolomics signal.</div>""", unsafe_allow_html=True)


class MetaRAPage:
    """Renders the MetaRA (metabolomics classifier) page."""

    def __init__(self, rf_a, top_f, XteAs, yteA):
        self.rf_a   = rf_a
        self.top_f  = top_f
        self.XteAs  = XteAs
        self.yteA   = yteA

    def render(self) -> None:
        st.markdown("<h1 style='color:#0D1B2A'>MetaRA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B7280'>Metabolomics classifier — predicts High vs Low DAS28 disease activity.</p>", unsafe_allow_html=True)
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", "0.654"); c2.metric("Test AUC", "0.676")
        c3.metric("CV AUC", "0.839");        c4.metric("Metabolites", "50 / 686")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='sec'>ROC Curve</div>", unsafe_allow_html=True)
            self._plot_roc()
        with col2:
            st.markdown("<div class='sec'>Top 15 Metabolite Importances</div>", unsafe_allow_html=True)
            self._plot_importances()

        st.markdown("---")
        self._render_upload_section()

    def _plot_roc(self) -> None:
        proba  = self.rf_a.predict_proba(self.XteAs)[:, 1]
        fpr, tpr, _ = roc_curve(self.yteA, proba)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#F5F7FA")
        ax.set_facecolor("#F5F7FA")
        ax.plot(fpr, tpr, color="#1B998B", lw=2.5, label=f"AUC={auc(fpr,tpr):.3f}")
        ax.fill_between(fpr, tpr, alpha=0.10, color="#1B998B")
        ax.plot([0,1],[0,1],"--",lw=1,color="#6B7280")
        ax.set_xlabel("FPR", color="#0D1B2A"); ax.set_ylabel("TPR", color="#0D1B2A")
        ax.set_title("MetaRA — ROC Curve", color="#0D1B2A", fontweight="bold")
        ax.legend(loc="lower right"); ax.tick_params(colors="#0D1B2A")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    def _plot_importances(self) -> None:
        imp_s = pd.Series(self.rf_a.feature_importances_, index=self.top_f).nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#F5F7FA")
        ax.set_facecolor("#F5F7FA")
        imp_s.plot(kind="barh", ax=ax,
                   color=["#E84855" if v == imp_s.max() else "#1B998B" for v in imp_s.values],
                   edgecolor="white")
        ax.set_title("Top 15 Metabolite Importances", color="#0D1B2A", fontweight="bold")
        ax.set_xlabel("Importance", color="#0D1B2A"); ax.tick_params(colors="#0D1B2A")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    def _render_upload_section(self) -> None:
        st.markdown("<div class='sec'>Predict from Patient Metabolomics File</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-note'>Upload a <b>CSV or TSV</b> with rows=metabolites, columns=patients (same format as Hur et al.).</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload metabolomics file", type=["csv","tsv"])
        if uploaded:
            self._predict_uploaded(uploaded)
        else:
            st.info("No file uploaded yet.")

    def _predict_uploaded(self, uploaded) -> None:
        try:
            sep = "\t" if uploaded.name.endswith(".tsv") else ","
            raw = pd.read_csv(uploaded, sep=sep, index_col=0)
            if raw.shape[0] > raw.shape[1]: raw = raw.T
            raw = raw.apply(pd.to_numeric, errors="coerce")
            available = [f for f in self.top_f if f in raw.columns]
            missing   = [f for f in self.top_f if f not in raw.columns]
            if len(available) < 10:
                st.error(f"Only {len(available)} metabolites found.")
                return
            if missing:
                st.warning(f"{len(missing)} metabolites missing — filled with median.")
            X_up = pd.DataFrame(0.0, index=raw.index, columns=self.top_f)
            for f in available: X_up[f] = raw[f].values
            for f in missing:   X_up[f] = raw[available].median(axis=1).values
            probas = self.rf_a.predict_proba(X_up.values)[:, 1]
            preds  = (probas >= 0.5).astype(int)
            st.success(f"Predicted {len(raw)} patients.")
            st.dataframe(pd.DataFrame({
                "Patient"       : raw.index,
                "Prediction"    : ["High DAS28" if p == 1 else "Low DAS28" for p in preds],
                "P(High DAS28)" : probas.round(4)
            }), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error: {e}")


class ClinicalRAPage:
    """Renders the ClinicalRA (clinical biomarker classifier) page."""

    def __init__(self, rf_b, cf):
        self.rf_b = rf_b
        self.cf   = cf

    def render(self) -> None:
        st.markdown("<h1 style='color:#0D1B2A'>ClinicalRA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B7280'>Clinical biomarker classifier — predicts RA vs Normal.</p>", unsafe_allow_html=True)
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", "1.000"); c2.metric("Test AUC", "1.000")
        c3.metric("Top Feature", "ESR (33%)"); c4.metric("Training Size", "3,561")
        st.markdown("---")

        st.markdown("<div class='sec'>Enter Patient Clinical Values</div>", unsafe_allow_html=True)
        inputs = self._render_input_form()
        proba_b, pred_b = self._predict(inputs)

        st.markdown("---")
        self._render_prediction(proba_b, pred_b)
        st.markdown("---")
        self._render_feature_importance()

    def _render_input_form(self) -> dict:
        """Render the input widgets and return raw values as a dict."""
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Continuous Markers**")
            age     = st.number_input("Age (years)",   min_value=18,  max_value=100, value=50,    step=1)
            esr     = st.number_input("ESR (mm/hr)",   min_value=0.0, max_value=150.0, value=29.0, step=0.5)
            crp     = st.number_input("CRP (mg/L)",    min_value=0.0, max_value=200.0, value=15.0, step=0.5)
            rf_val  = st.number_input("RF (IU/mL)",    min_value=0.0, max_value=500.0, value=25.0, step=0.5)
            anticcp = st.number_input("Anti-CCP (U/mL)", min_value=0.0, max_value=500.0, value=25.0, step=0.5)
            c3      = st.number_input("C3 (mg/dL)",    min_value=50.0, max_value=300.0, value=142.0, step=1.0)
            c4_val  = st.number_input("C4 (mg/dL)",    min_value=5.0,  max_value=100.0, value=42.0,  step=1.0)
        with col2:
            st.markdown("**Categorical Markers**")
            gender  = st.selectbox("Gender",     ["Male", "Female"])
            hlab27  = st.selectbox("HLA-B27",    ["Negative", "Positive"])
            ana     = st.selectbox("ANA",        ["Negative", "Positive"])
            anti_ro = st.selectbox("Anti-Ro",    ["Negative", "Positive"])
            anti_la = st.selectbox("Anti-La",    ["Negative", "Positive"])
            anti_ds = st.selectbox("Anti-dsDNA", ["Negative", "Positive"])
            anti_sm = st.selectbox("Anti-Sm",    ["Negative", "Positive"])
        return dict(
            age=age, esr=esr, crp=crp, rf_val=rf_val, anticcp=anticcp,
            c3=c3, c4_val=c4_val, gender=gender, hlab27=hlab27,
            ana=ana, anti_ro=anti_ro, anti_la=anti_la,
            anti_ds=anti_ds, anti_sm=anti_sm
        )

    def _predict(self, inputs: dict) -> tuple[float, int]:
        """Build model input from form values and return (probability, prediction)."""
        enc = {"Positive": 1, "Negative": 0, "Male": 0, "Female": 1}
        input_df = pd.DataFrame([{
            "Age": inputs["age"], "Gender": enc[inputs["gender"]],
            "ESR": inputs["esr"], "CRP": inputs["crp"],
            "RF": inputs["rf_val"], "Anti-CCP": inputs["anticcp"],
            "HLA-B27": enc[inputs["hlab27"]], "ANA": enc[inputs["ana"]],
            "Anti-Ro": enc[inputs["anti_ro"]], "Anti-La": enc[inputs["anti_la"]],
            "Anti-dsDNA": enc[inputs["anti_ds"]], "Anti-Sm": enc[inputs["anti_sm"]],
            "C3": inputs["c3"], "C4": inputs["c4_val"]
        }])[self.cf]   # reorder to match training column order
        proba = self.rf_b.predict_proba(input_df.values)[0][1]
        return float(proba), int(proba >= 0.5)

    def _render_prediction(self, proba: float, pred: int) -> None:
        st.markdown("<div class='sec'>Prediction Result</div>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])
        with r1:
            if pred == 1:
                st.markdown(f"<div class='pred-ra'><div class='pred-title'>⚠ RA Detected</div><div class='pred-score'>P(RA) = {proba:.4f}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='pred-normal'><div class='pred-title'>✓ Normal</div><div class='pred-score'>P(RA) = {proba:.4f}</div></div>", unsafe_allow_html=True)
        with r2:
            self._plot_probability_bar(proba)

    def _plot_probability_bar(self, proba: float) -> None:
        fig, ax = plt.subplots(figsize=(6, 1.1), facecolor="#F5F7FA")
        ax.set_facecolor("#F5F7FA")
        bc = "#E84855" if proba >= 0.5 else "#1B998B"
        ax.barh(0, proba, color=bc, height=0.5)
        ax.barh(0, 1.0 - proba, left=proba, color="#E2E8F0", height=0.5)
        ax.axvline(0.5, color="#0D1B2A", lw=2, linestyle="--")
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0","0.25","0.5","0.75","1.0"], color="#0D1B2A", fontsize=9)
        ax.set_title("P(RA) Probability", color="#0D1B2A", fontsize=11, fontweight="bold")
        ax.spines[["top","right","left"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    def _render_feature_importance(self) -> None:
        st.markdown("<div class='sec'>Feature Importances</div>", unsafe_allow_html=True)
        imp_b = pd.Series(self.rf_b.feature_importances_, index=self.cf).sort_values()
        colors = [
            "#E84855" if v == imp_b.max()
            else "#F4A261" if v >= imp_b.quantile(0.75)
            else "#1B998B"
            for v in imp_b.values
        ]
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F5F7FA")
        ax.set_facecolor("#F5F7FA")
        imp_b.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_title("ClinicalRA — Feature Importances", color="#0D1B2A", fontweight="bold")
        ax.set_xlabel("Importance", color="#0D1B2A"); ax.tick_params(colors="#0D1B2A")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()


class FusionRAPage:
    """Renders the FusionRA (late fusion) page."""

    def __init__(self, rf_a, top_f, rf_b, cf, XteB, yteB, proba_a_all):
        self.rf_a        = rf_a
        self.top_f       = top_f
        self.rf_b        = rf_b
        self.cf          = cf
        self.XteB        = XteB
        self.yteB        = yteB
        self.proba_a_all = proba_a_all

    def render(self) -> None:
        st.markdown("<h1 style='color:#0D1B2A'>FusionRA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B7280'>Upload metabolomics data, enter clinical values, then adjust weights.</p>", unsafe_allow_html=True)
        st.markdown("---")

        score_a = self._render_metabolomics_upload()
        score_b = self._render_clinical_inputs()
        w_a     = self._render_weight_slider()
        self._render_fused_prediction(score_a, score_b, w_a)
        st.markdown("---")
        self._render_population_analysis(w_a)

    def _render_metabolomics_upload(self) -> np.ndarray | None:
        st.markdown("<div class='sec'>Step 1 — Upload Metabolomics File (MetaRA)</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload metabolomics file (.csv or .tsv)", type=["csv","tsv"], key="fus_met")
        if not uploaded:
            st.info("Upload a metabolomics file to get MetaRA scores.")
            return None
        try:
            sep       = "\t" if uploaded.name.endswith(".tsv") else ","
            raw       = pd.read_csv(uploaded, sep=sep, index_col=0)
            if raw.shape[0] > raw.shape[1]: raw = raw.T
            raw       = raw.apply(pd.to_numeric, errors="coerce")
            available = [f for f in self.top_f if f in raw.columns]
            missing   = [f for f in self.top_f if f not in raw.columns]
            if len(available) < 10:
                st.error(f"Only {len(available)} metabolites found."); return None
            if missing:
                st.warning(f"{len(missing)} metabolites missing — filled with median.")
            X_up = pd.DataFrame(0.0, index=raw.index, columns=self.top_f)
            for f in available: X_up[f] = raw[f].values
            for f in missing:   X_up[f] = raw[available].median(axis=1).values
            scores = self.rf_a.predict_proba(X_up.values)[:, 1]
            st.success(f"MetaRA: {len(raw)} patient(s) — mean P(High DAS28): {scores.mean():.4f}")
            return scores
        except Exception as e:
            st.error(f"Error: {e}"); return None

    def _render_clinical_inputs(self) -> float:
        st.markdown("---")
        st.markdown("<div class='sec'>Step 2 — Enter Clinical Values (ClinicalRA)</div>", unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Continuous Markers**")
            f_age    = st.number_input("Age",         min_value=18,  max_value=100,   value=50,    step=1,   key="f_age")
            f_esr    = st.number_input("ESR (mm/hr)", min_value=0.0, max_value=150.0, value=29.0,  step=0.5, key="f_esr")
            f_crp    = st.number_input("CRP (mg/L)",  min_value=0.0, max_value=200.0, value=15.0,  step=0.5, key="f_crp")
            f_rf     = st.number_input("RF (IU/mL)",  min_value=0.0, max_value=500.0, value=25.0,  step=0.5, key="f_rf")
            f_anticcp= st.number_input("Anti-CCP",    min_value=0.0, max_value=500.0, value=25.0,  step=0.5, key="f_anticcp")
            f_c3     = st.number_input("C3 (mg/dL)",  min_value=50.0,max_value=300.0, value=142.0, step=1.0, key="f_c3")
            f_c4     = st.number_input("C4 (mg/dL)",  min_value=5.0, max_value=100.0, value=42.0,  step=1.0, key="f_c4")
        with fc2:
            st.markdown("**Categorical Markers**")
            f_gender = st.selectbox("Gender",     ["Male","Female"],              key="f_gender")
            f_hlab27 = st.selectbox("HLA-B27",    ["Negative","Positive"],        key="f_hlab27")
            f_ana    = st.selectbox("ANA",         ["Negative","Positive"],        key="f_ana")
            f_anti_ro= st.selectbox("Anti-Ro",    ["Negative","Positive"],        key="f_anti_ro")
            f_anti_la= st.selectbox("Anti-La",    ["Negative","Positive"],        key="f_anti_la")
            f_anti_ds= st.selectbox("Anti-dsDNA", ["Negative","Positive"],        key="f_anti_ds")
            f_anti_sm= st.selectbox("Anti-Sm",    ["Negative","Positive"],        key="f_anti_sm")

        enc = {"Positive":1,"Negative":0,"Male":0,"Female":1}
        input_df = pd.DataFrame([{
            "Age": f_age, "Gender": enc[f_gender], "ESR": f_esr, "CRP": f_crp,
            "RF": f_rf, "Anti-CCP": f_anticcp, "HLA-B27": enc[f_hlab27],
            "ANA": enc[f_ana], "Anti-Ro": enc[f_anti_ro], "Anti-La": enc[f_anti_la],
            "Anti-dsDNA": enc[f_anti_ds], "Anti-Sm": enc[f_anti_sm],
            "C3": f_c3, "C4": f_c4
        }])[self.cf]
        score_b = float(self.rf_b.predict_proba(input_df.values)[0][1])
        st.markdown(f"<div class='info-note'><b>ClinicalRA score:</b> P(RA) = <b>{score_b:.4f}</b></div>", unsafe_allow_html=True)
        return score_b

    def _render_weight_slider(self) -> float:
        st.markdown("---")
        st.markdown("<div class='sec'>Step 3 — Adjust Weights & Get Fused Prediction</div>", unsafe_allow_html=True)
        col_sl, col_pre = st.columns([3, 2])
        with col_pre:
            preset_sel = st.selectbox("Or pick a preset:", list(DashboardConfig.PRESETS.keys()))
            if st.button("Apply Preset"):
                st.session_state["fusion_w_a"] = DashboardConfig.PRESETS[preset_sel]
                st.rerun()
        with col_sl:
            w_a = st.slider(
                "ClinicalRA weight ← → MetaRA weight",
                min_value=0.0, max_value=1.0, step=0.05,
                value=st.session_state.get("fusion_w_a", 0.30),
                key="fusion_slider"
            )
            st.session_state["fusion_w_a"] = w_a
        w_b = round(1.0 - w_a, 2)
        c1, c2, c3 = st.columns(3)
        c1.metric("MetaRA Weight", f"{w_a:.2f}")
        c2.metric("ClinicalRA Weight", f"{w_b:.2f}")
        c3.metric("Sum", "1.00")
        return w_a

    def _render_fused_prediction(self, score_a: np.ndarray | None, score_b: float, w_a: float) -> None:
        if score_a is None:
            st.markdown("<div class='warn'>Upload a metabolomics file in Step 1 to get the fused prediction.</div>", unsafe_allow_html=True)
            return
        w_b      = round(1.0 - w_a, 2)
        mean_sa  = float(score_a.mean())
        fused_p  = w_a * mean_sa + w_b * score_b
        st.markdown("---")
        st.markdown("<div class='sec'>Fused Prediction</div>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.metric("MetaRA Score",    f"{mean_sa:.4f}",  "P(High DAS28)")
        p2.metric("ClinicalRA Score",f"{score_b:.4f}",  "P(RA)")
        p3.metric("Fused Score",     f"{fused_p:.4f}",  "Combined")
        st.markdown("<br>", unsafe_allow_html=True)
        if int(fused_p >= 0.5) == 1:
            st.markdown(f"<div class='pred-ra'><div class='pred-title'>⚠ FusionRA — RA Indicated</div><div class='pred-score'>Score={fused_p:.4f} | MetaRA({w_a:.2f}) + ClinicalRA({w_b:.2f})</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='pred-normal'><div class='pred-title'>✓ FusionRA — Normal Indicated</div><div class='pred-score'>Score={fused_p:.4f} | MetaRA({w_a:.2f}) + ClinicalRA({w_b:.2f})</div></div>", unsafe_allow_html=True)

    def _render_population_analysis(self, w_a: float) -> None:
        st.markdown("<div class='sec'>Population-Level Analysis (Test Set)</div>", unsafe_allow_html=True)
        rng      = np.random.default_rng(DashboardConfig.RANDOM_STATE)
        sa_pop   = rng.choice(self.proba_a_all, size=len(self.yteB), replace=True)
        sb_pop   = self.rf_b.predict_proba(self.XteB)[:, 1]
        fp       = w_a * sa_pop + (1 - w_a) * sb_pop
        auc_f    = roc_auc_score(self.yteB, fp)
        auc_b    = roc_auc_score(self.yteB, sb_pop)
        col1, col2 = st.columns(2)
        with col1:
            self._plot_roc_comparison(sb_pop, fp, auc_b, auc_f, w_a)
        with col2:
            self._plot_weight_sensitivity(sa_pop, sb_pop, auc_b, w_a)

    def _plot_roc_comparison(self, sb_pop, fp, auc_b, auc_f, w_a) -> None:
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#F5F7FA"); ax.set_facecolor("#F5F7FA")
        for lbl, ys, col, ls in [
            (f"ClinicalRA (AUC={auc_b:.3f})", sb_pop, "#1B998B", "-"),
            (f"FusionRA w_A={w_a:.2f} (AUC={auc_f:.3f})", fp, "#E84855", "--")
        ]:
            fpr, tpr, _ = roc_curve(self.yteB, ys)
            ax.plot(fpr, tpr, lw=2.5, label=lbl, color=col, linestyle=ls)
        ax.plot([0,1],[0,1],"--",lw=1,color="#6B7280")
        ax.set_xlabel("FPR",color="#0D1B2A"); ax.set_ylabel("TPR",color="#0D1B2A")
        ax.set_title("ROC Curve Comparison",color="#0D1B2A",fontweight="bold")
        ax.legend(loc="lower right",fontsize=9); ax.tick_params(colors="#0D1B2A")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    def _plot_weight_sensitivity(self, sa_pop, sb_pop, auc_b, w_a) -> None:
        ws   = np.arange(0.0, 1.05, 0.05)
        aucs = [roc_auc_score(self.yteB, w * sa_pop + (1-w) * sb_pop) for w in ws]
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#F5F7FA"); ax.set_facecolor("#F5F7FA")
        ax.plot(ws, aucs, color="#1B998B", lw=2.5, marker="o", markersize=4)
        ax.axvline(w_a, color="#E84855", lw=2, linestyle="--", label=f"w_A={w_a:.2f}")
        ax.axhline(auc_b, color="#6B7280", lw=1, linestyle=":", label=f"ClinicalRA={auc_b:.4f}")
        ax.set_xlabel("MetaRA Weight",color="#0D1B2A"); ax.set_ylabel("Fused AUC",color="#0D1B2A")
        ax.set_title("Weight Sensitivity",color="#0D1B2A",fontweight="bold")
        ax.legend(fontsize=9); ax.tick_params(colors="#0D1B2A")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()


# =============================================================================
# Dashboard  — top-level controller
# =============================================================================
class Dashboard:
    """
    Top-level controller.
    Initialises the app, applies CSS, renders the sidebar,
    and routes to the correct page class.
    """

    NAV_OPTIONS = ["Home", "MetaRA", "ClinicalRA", "FusionRA"]

    def __init__(self):
        st.set_page_config(
            page_title="FusionRA", page_icon="🧬",
            layout="wide", initial_sidebar_state="expanded"
        )

    def _apply_styles(self) -> None:
        st.markdown(DashboardConfig.CSS, unsafe_allow_html=True)

    def _render_sidebar(self) -> str:
        with st.sidebar:
            st.markdown("<h2 style='color:#FFFFFF; margin-bottom:4px'>FusionRA</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color:#B0C8E8; font-size:14px; margin-top:0'>RA Multi-Modal ML Pipeline</p>", unsafe_allow_html=True)
            st.markdown("---")
            default_idx = self.NAV_OPTIONS.index(st.session_state.get("nav", "Home"))
            page = st.radio("Navigate", self.NAV_OPTIONS,
                            index=default_idx, label_visibility="collapsed")
            st.session_state["nav"] = page
            st.markdown("---")
            st.markdown("""
            <div style='line-height:2.2'>
                <span style='color:#FFFFFF; font-weight:700; font-size:15px'>MetaRA</span>
                <span style='color:#C8DCF0; font-size:14px'> — Metabolomics</span><br>
                <span style='color:#FFFFFF; font-weight:700; font-size:15px'>ClinicalRA</span>
                <span style='color:#C8DCF0; font-size:14px'> — Clinical</span><br>
                <span style='color:#FFFFFF; font-weight:700; font-size:15px'>FusionRA</span>
                <span style='color:#C8DCF0; font-size:14px'> — Combined</span>
            </div>""", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("""
            <div style='line-height:2.2'>
                <span style='color:#FFFFFF; font-weight:700; font-size:15px'>Sources</span><br>
                <span style='color:#C8DCF0; font-size:14px'>Hur et al. 2021</span><br>
                <span style='color:#C8DCF0; font-size:14px'>Mahdi et al. 2025</span><br><br>
                <span style='color:#FFFFFF; font-weight:700; font-size:15px'>Stack</span><br>
                <span style='color:#C8DCF0; font-size:14px'>Python · scikit-learn · Streamlit</span>
            </div>""", unsafe_allow_html=True)
        return page

    def run(self) -> None:
        """Entry point — apply styles, load models, route to page."""
        self._apply_styles()

        with st.spinner("Loading models..."):
            (rf_a, top_f, tidx, mf,
             XteAs, yteA,
             rf_b, cf, XteB, yteB,
             proba_a_all) = ModelCache.train_models()

        page = self._render_sidebar()

        if page == "Home":
            HomePage().render(models=None)

        elif page == "MetaRA":
            MetaRAPage(rf_a, top_f, XteAs, yteA).render()

        elif page == "ClinicalRA":
            ClinicalRAPage(rf_b, cf).render()

        elif page == "FusionRA":
            FusionRAPage(rf_a, top_f, rf_b, cf, XteB, yteB, proba_a_all).render()


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    Dashboard().run()
