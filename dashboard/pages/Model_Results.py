"""
Model Results
=============
Page Streamlit : comparaison des performances des modèles de détection de fraude.
Compatible avec les splits: data/train.parquet, data/val.parquet, data/test.parquet
et les modèles *.pkl dans models/ (RF / XGB / LGBM / CatBoost).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# Page config + style léger "projet"
# ============================================================
st.set_page_config(
    page_title="Model Results - Fraud Detection",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem; }
      .sub-title { color: #6b7280; margin-top: 0; margin-bottom: 1.2rem; }
      .small { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Model Results</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Comparaison des performances des modèles de détection de fraude sur le jeu de test.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # repo/
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# ============================================================
# Utils: list/load models, expected features, align, proba safe
# ============================================================
def list_model_files(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.glob("*.pkl")])


@st.cache_resource
def load_model(models_dir: Path, filename: str):
    path = models_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {path}")
    return joblib.load(path)


def get_expected_feature_names(model, X_fallback: pd.DataFrame) -> list[str]:
    """
    Essaie de récupérer les noms de features attendus par le modèle.
    - sklearn: feature_names_in_
    - CatBoost: feature_names_
    - LightGBM: feature_name_
    - XGBoost: model.get_booster().feature_names
    Sinon: fallback = X_fallback.columns
    """
    if hasattr(model, "feature_names_in_"):
        try:
            fn = list(model.feature_names_in_)
            if fn:
                return fn
        except Exception:
            pass

    if hasattr(model, "feature_names_"):  # catboost
        try:
            fn = list(model.feature_names_)
            if fn:
                return fn
        except Exception:
            pass

    if hasattr(model, "feature_name_"):  # lightgbm
        try:
            fn = list(model.feature_name_)
            if fn:
                return fn
        except Exception:
            pass

    if hasattr(model, "get_booster"):  # xgboost
        try:
            bn = model.get_booster().feature_names
            if bn:
                return list(bn)
        except Exception:
            pass

    return list(X_fallback.columns)


def align_X_to_features(X: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Ajoute colonnes manquantes à 0, supprime les extras, impose l'ordre exact."""
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = 0
    return X2[expected].copy()


def safe_predict_proba_pos(model, X: pd.DataFrame) -> np.ndarray:
    """Retourne p(classe=1) de manière robuste."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
        raise ValueError("predict_proba a renvoyé une forme inattendue.")

    # fallback: parfois predict renvoie un score/proba
    pred = np.asarray(model.predict(X))
    if pred.ndim == 2 and pred.shape[1] >= 2:
        return pred[:, 1]
    if pred.ndim == 1:
        return pred
    raise ValueError("Impossible d'obtenir une probabilité depuis ce modèle.")


def model_display_name(filename: str) -> str:
    """Nom lisible depuis le fichier."""
    name = filename.replace(".pkl", "")
    name = name.replace("_", " ").replace("-", " ")
    return name.title()


# ============================================================
# Load data splits
# ============================================================
@st.cache_data
def load_splits(data_dir: Path):
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"

    missing = [p.name for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        return None, None, None, f"Manquants dans {data_dir}: {missing}"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    return train_df, val_df, test_df, None


train_df, val_df, test_df_full, err = load_splits(DATA_DIR)
if err:
    st.error("Impossible de charger les données.")
    st.info("Attendu dans /data : train.parquet, val.parquet, test.parquet")
    st.code(err)
    st.stop()

# ============================================================
# Sidebar: target + model selection
# ============================================================
st.sidebar.header("Paramètres")

# target selection
binary_cols = [c for c in train_df.columns if train_df[c].dropna().nunique() == 2]
all_cols = list(train_df.columns)

if not all_cols:
    st.error("Dataset vide ou colonnes non détectées.")
    st.stop()

target_col = st.sidebar.selectbox(
    "Colonne cible (fraude)",
    options=binary_cols + [c for c in all_cols if c not in binary_cols],
    index=0 if binary_cols else 0,
)

# models selection
model_files = list_model_files(MODELS_DIR)
if not model_files:
    st.error("Aucun modèle .pkl trouvé dans /models.")
    st.stop()

selected_model_file = st.sidebar.selectbox("Modèle (détail)", model_files)

view_mode = st.sidebar.radio("Mode d'affichage", ["Comparaison globale", "Détail d'un modèle"], index=1)

# ============================================================
# Prepare X_test / y_test + base feature alignment train/test
# ============================================================
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df_full.drop(columns=[target_col])
y_test = test_df_full[target_col]
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]
y_test = np.asarray(y_test).ravel()

common_cols = [c for c in X_test_raw.columns if c in X_train_raw.columns]
X_train_common = X_train_raw[common_cols].copy()
X_test_common = X_test_raw[common_cols].copy()

if X_test_common.shape[1] == 0:
    st.error("Aucune feature commune entre train et test après suppression de la cible.")
    st.stop()

# ============================================================
# Load all models & evaluate
# ============================================================
@st.cache_data
def evaluate_models(model_filenames: list[str], X_train_common: pd.DataFrame, X_test_common: pd.DataFrame, y_test: np.ndarray):
    results = {}
    roc_data = {}

    for fn in model_filenames:
        name = model_display_name(fn)

        try:
            mdl = load_model(MODELS_DIR, fn)

            # align per-model (CatBoost/LGBM/XGB strict)
            expected = get_expected_feature_names(mdl, X_train_common)
            if not expected:
                expected = list(X_train_common.columns)

            X_test_aligned = align_X_to_features(X_test_common, expected)

            # predictions
            proba = safe_predict_proba_pos(mdl, X_test_aligned)
            pred = (proba >= 0.5).astype(int)

            # metrics
            cm = confusion_matrix(y_test, pred)
            auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) == 2 else np.nan

            # roc curve
            try:
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
            except Exception:
                pass

            # feature importance (best effort)
            importance = None
            if hasattr(mdl, "feature_importances_"):
                imp = np.asarray(getattr(mdl, "feature_importances_"))
                if imp.ndim == 1 and len(imp) == len(expected):
                    importance = dict(zip(expected, imp))
            elif hasattr(mdl, "coef_"):
                coef = np.asarray(getattr(mdl, "coef_"))
                if coef.ndim >= 1:
                    coef = np.abs(coef.ravel())
                    if len(coef) == len(expected):
                        importance = dict(zip(expected, coef))

            results[name] = {
                "file": fn,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": auc,
                "confusion_matrix": cm,
                "y_pred": pred,
                "y_proba": proba,
                "feature_importance": importance,  # peut être None
                "expected_features": expected,
            }

        except Exception as e:
            # on garde une trace d'erreur pour affichage
            results[name] = {"error": str(e), "file": fn}

    return results, roc_data


with st.spinner("Évaluation des modèles…"):
    results, roc_data = evaluate_models(model_files, X_train_common, X_test_common, y_test)

# filtrer ceux qui ont des erreurs
valid_models = [k for k, v in results.items() if "error" not in v]
error_models = {k: v for k, v in results.items() if "error" in v}

if not valid_models:
    st.error("Aucun modèle n'a pu être évalué.")
    if error_models:
        st.markdown("### Erreurs rencontrées")
        for k, v in error_models.items():
            st.warning(f"**{k}** ({v.get('file','?')}): {v.get('error')}")
    st.stop()

# metrics top row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modèles évalués", f"{len(valid_models)}/{len(model_files)}")
c2.metric("Features (test)", f"{X_test_common.shape[1]}")
c3.metric("Taux fraude (réel)", f"{float((y_test==1).mean()):.1%}")
best_model_name = max(valid_models, key=lambda m: results[m]["f1"])
c4.metric("Meilleur F1", f"{results[best_model_name]['f1']:.2%}", delta=best_model_name)

if error_models:
    with st.expander("Modèles ignorés (erreurs)"):
        for k, v in error_models.items():
            st.warning(f"**{k}** ({v.get('file','?')}): {v.get('error')}")

st.markdown("---")

# ============================================================
# View 1: Global comparison
# ============================================================
if view_mode == "Comparaison globale":
    st.markdown("## Comparaison des modèles")

    comparison_rows = []
    for name in valid_models:
        m = results[name]
        comparison_rows.append(
            {
                "Modèle": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1"],
                "ROC-AUC": m["roc_auc"],
            }
        )

    comp_df = pd.DataFrame(comparison_rows).sort_values("F1-Score", ascending=False)

    st.markdown("### Tableau récapitulatif")
    show_df = comp_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        show_df[col] = show_df[col].map(lambda x: f"{x:.2%}")
    show_df["ROC-AUC"] = show_df["ROC-AUC"].map(lambda x: "N/A" if np.isnan(x) else f"{x:.3f}")

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.success(f"**Meilleur modèle (F1-Score):** {comp_df.iloc[0]['Modèle']} — {comp_df.iloc[0]['F1-Score']:.2%}")

    st.markdown("### Visualisation des métriques")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig = go.Figure()
    for name in comp_df["Modèle"].tolist():
        row = comp_df[comp_df["Modèle"] == name].iloc[0]
        fig.add_trace(
            go.Bar(
                name=name,
                x=metrics_to_plot,
                y=[row[m] if not (m == "ROC-AUC" and np.isnan(row[m])) else 0 for m in metrics_to_plot],
            )
        )
    fig.update_layout(
        title="Comparaison des métriques",
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Courbes ROC")
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC=0.5)", line=dict(color="gray", dash="dash"))
    )

    for name in valid_models:
        if name in roc_data:
            d = roc_data[name]
            auc = d.get("auc", np.nan)
            label = f"{name} (AUC={auc:.3f})" if not np.isnan(auc) else f"{name} (AUC=N/A)"
            fig_roc.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=label))

    fig_roc.update_layout(
        title="ROC — comparaison",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        height=520,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ============================================================
# View 2: Model detail
# ============================================================
else:
    st.markdown("## Détail d'un modèle")

    selected_display = model_display_name(selected_model_file)
    if selected_display not in results or "error" in results[selected_display]:
        st.error("Le modèle sélectionné n'a pas pu être évalué.")
        st.info("Choisis un autre modèle (ou regarde les erreurs dans l'expander).")
        st.stop()

    m = results[selected_display]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{m['accuracy']:.2%}")
    col2.metric("Precision", f"{m['precision']:.2%}")
    col3.metric("Recall", f"{m['recall']:.2%}")
    col4.metric("F1-Score", f"{m['f1']:.2%}")
    col5.metric("ROC-AUC", "N/A" if np.isnan(m["roc_auc"]) else f"{m['roc_auc']:.3f}")

    st.markdown("---")
    st.markdown("### Matrice de confusion")

    cm = m["confusion_matrix"]
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Prédit", y="Réel", color="Nombre"),
        x=["Non-Fraude", "Fraude"],
        y=["Non-Fraude", "Fraude"],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(title=f"Matrice de confusion — {selected_display}")
    st.plotly_chart(fig_cm, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    fpr = fp / (tn + fp) if (tn + fp) else 0

    st.info(
        f"""
**Détails**
- True Negatives (TN): {tn:,}
- False Positives (FP): {fp:,} (taux de fausse alerte: {fpr:.2%})
- False Negatives (FN): {fn:,}
- True Positives (TP): {tp:,}
- Spécificité (TNR): {specificity:.2%}
"""
    )

    st.markdown("### ROC (modèle)")
    if selected_display in roc_data:
        d = roc_data[selected_display]
        auc = d.get("auc", np.nan)
        fig_roc_single = go.Figure()
        fig_roc_single.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC=0.5)", line=dict(color="gray", dash="dash"))
        )
        fig_roc_single.add_trace(
            go.Scatter(
                x=d["fpr"],
                y=d["tpr"],
                mode="lines",
                name=f"{selected_display} (AUC={auc:.3f})" if not np.isnan(auc) else f"{selected_display} (AUC=N/A)",
                fill="tozeroy",
            )
        )
        fig_roc_single.update_layout(
            title=f"ROC — {selected_display}",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            height=420,
        )
        st.plotly_chart(fig_roc_single, use_container_width=True)
    else:
        st.warning("ROC indisponible pour ce modèle.")

    st.markdown("### Feature importance")
    if m["feature_importance"] is None:
        st.warning("Feature importance indisponible pour ce modèle (pas d'attribut importances/coef).")
    else:
        imp = pd.DataFrame(
            {"Feature": list(m["feature_importance"].keys()), "Importance": list(m["feature_importance"].values())}
        ).sort_values("Importance", ascending=False)

        topk = st.slider("Top features à afficher", 10, min(50, len(imp)), 20)
        imp_top = imp.head(topk).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            imp_top,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top {topk} features — {selected_display}",
        )
        fig_imp.update_layout(height=520)
        st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# Exports
# ============================================================
st.markdown("---")
st.markdown("## Exporter les résultats")

export_rows = []
for name in valid_models:
    m = results[name]
    export_rows.append(
        {
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC_AUC": m["roc_auc"],
        }
    )
export_df = pd.DataFrame(export_rows).sort_values("F1", ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="Télécharger comparaison (CSV)",
        data=export_df.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

with col2:
    # Feature importance export (si dispo)
    all_imp_rows = []
    for name in valid_models:
        imp = results[name].get("feature_importance")
        if not imp:
            continue
        for feat, val in imp.items():
            all_imp_rows.append({"Model": name, "Feature": feat, "Importance": val})

    if all_imp_rows:
        imp_df = pd.DataFrame(all_imp_rows)
        st.download_button(
            label="Télécharger feature importance (CSV)",
            data=imp_df.to_csv(index=False),
            file_name="feature_importance.csv",
            mime="text/csv",
        )
    else:
        st.caption("Feature importance non disponible (aucun modèle ne fournit d'importances/coef).")

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: gray; font-size: 0.9rem;">
  <p>Résultats calculés à partir des modèles sauvegardés et du jeu de test.</p>
</div>
""",
    unsafe_allow_html=True,
)
