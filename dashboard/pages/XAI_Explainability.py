from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from xai_utils import (
    SHAPExplainer,
    LIMEExplainer,
    compare_shap_lime,
    safe_predict_proba_pos,
)

# ============================================================
# Page config & style
# ============================================================
st.set_page_config(
    page_title="XAI Explainability - Fraud Detection",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title { font-size: 2.3rem; font-weight: 800; margin-bottom: 0.2rem; }
      .sub-title { color: #6b7280; margin-top: 0; margin-bottom: 1.2rem; }
      .card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.02);
      }
      .small { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">XAI Explainability</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Comprendre les décisions des modèles de détection de fraude via SHAP (global/local) et LIME.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# ============================================================
# Helpers: expected features per model + alignment
# ============================================================
def get_expected_feature_names(model, X_fallback: pd.DataFrame) -> list[str]:
    # sklearn
    if hasattr(model, "feature_names_in_"):
        try:
            fn = list(model.feature_names_in_)
            if fn:
                return fn
        except Exception:
            pass

    # catboost
    if hasattr(model, "feature_names_"):
        try:
            fn = list(model.feature_names_)
            if fn:
                return fn
        except Exception:
            pass

    # lightgbm
    if hasattr(model, "feature_name_"):
        try:
            fn = list(model.feature_name_)
            if fn:
                return fn
        except Exception:
            pass

    # xgboost
    if hasattr(model, "get_booster"):
        try:
            bn = model.get_booster().feature_names
            if bn:
                return list(bn)
        except Exception:
            pass

    return list(X_fallback.columns)


def align_X_to_features(X: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = 0
    return X2[expected].copy()


# ============================================================
# Loaders
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


@st.cache_resource
def load_model(models_dir: Path, model_file: str):
    path = models_dir / model_file
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {path}")
    return joblib.load(path)


def list_model_files(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.glob("*.pkl")])


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Paramètres")

model_files = list_model_files(MODELS_DIR)
if not model_files:
    st.sidebar.error("Aucun modèle .pkl trouvé dans le dossier /models.")
    st.stop()

selected_model_file = st.sidebar.selectbox("Modèle", model_files)

view = st.sidebar.radio(
    "Vue",
    ["SHAP Global", "SHAP Local", "LIME", "Comparaison"],
    index=0,
)

st.sidebar.markdown("---")

# ============================================================
# Load data
# ============================================================
train_df, val_df, test_df_full, err = load_splits(DATA_DIR)
if err:
    st.error("Impossible de charger les données.")
    st.info("Attendu dans /data : train.parquet, val.parquet, test.parquet")
    st.code(err)
    st.stop()

# ============================================================
# Target selection
# ============================================================
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

# Split X/y
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df_full.drop(columns=[target_col])
y_test = test_df_full[target_col]
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

# Features communes (sécurité)
common_cols = [c for c in X_test_raw.columns if c in X_train_raw.columns]
X_train = X_train_raw[common_cols].copy()
X_test = X_test_raw[common_cols].copy()

if X_train.shape[1] == 0:
    st.error("Aucune feature commune entre train et test après suppression de la cible.")
    st.stop()

# ============================================================
# Load model
# ============================================================
try:
    model = load_model(MODELS_DIR, selected_model_file)
except Exception as e:
    st.error("Impossible de charger le modèle.")
    st.exception(e)
    st.stop()

# ============================================================
# Align features to model (CRITICAL for CatBoost / LGBM / XGB)
# ============================================================
expected_features = get_expected_feature_names(model, X_train)
if not expected_features:
    expected_features = list(X_train.columns)

X_train = align_X_to_features(X_train, expected_features)
X_test = align_X_to_features(X_test, expected_features)
feature_names = list(X_test.columns)

# ============================================================
# Build scored test dataframe
# ============================================================
@st.cache_data
def make_test_df(_model, X_test_df: pd.DataFrame, y_test_s: pd.Series):
    proba = safe_predict_proba_pos(_model, X_test_df)

    # prédiction robuste
    try:
        pred = np.asarray(_model.predict(X_test_df)).ravel()
        if not set(np.unique(pred)).issubset({0, 1}):
            pred = (proba >= 0.5).astype(int)
        else:
            pred = pred.astype(int)
    except Exception:
        pred = (proba >= 0.5).astype(int)

    df = X_test_df.copy()
    df["fraud_probability"] = proba
    df["prediction"] = pred
    df["actual"] = np.asarray(y_test_s).ravel()
    df["transaction_id"] = [f"TXN_{i:06d}" for i in range(len(df))]
    return df


test_df = make_test_df(model, X_test, y_test)

# ============================================================
# Header metrics
# ============================================================
pred_fraud_rate = float((test_df["prediction"] == 1).mean())
true_fraud_rate = float((test_df["actual"] == 1).mean())
avg_proba = float(test_df["fraud_probability"].mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Modèle", selected_model_file.replace(".pkl", ""))
m2.metric("Cible", target_col)
m3.metric("Fraudes prédites", f"{pred_fraud_rate:.1%}")
m4.metric("Fraudes réelles", f"{true_fraud_rate:.1%}")

st.markdown("---")

# ============================================================
# Explainers (no cache: avoid stale feature sets)
# ============================================================
def get_shap_explainer(_model, _X_bg: pd.DataFrame, _feature_names: list[str]):
    return SHAPExplainer(_model, _X_bg, feature_names=_feature_names)

def get_lime_explainer(_model, _X_train: pd.DataFrame, _feature_names: list[str]):
    return LIMEExplainer(_model, _X_train, feature_names=_feature_names)


# ============================================================
# Views
# ============================================================
if view == "SHAP Global":
    st.markdown("### SHAP — Analyse globale")
    st.markdown(
        '<div class="small">Les SHAP values mesurent la contribution moyenne de chaque feature à la prédiction.</div>',
        unsafe_allow_html=True,
    )

    max_samples = st.sidebar.slider(
        "Échantillons expliqués (SHAP)",
        200,
        min(5000, len(X_test)),
        min(1000, len(X_test)),
        step=200,
    )
    bg_size = st.sidebar.slider(
        "Taille background (SHAP)",
        100,
        min(5000, len(X_train)),
        min(1000, len(X_train)),
        step=100,
    )
    top_k = st.sidebar.slider(
        "Top features",
        5,
        min(40, len(feature_names)),
        min(20, len(feature_names)),
    )

    X_bg = X_train.sample(n=min(bg_size, len(X_train)), random_state=42)
    explainer = get_shap_explainer(model, X_bg, feature_names)

    with st.spinner("Calcul des SHAP values…"):
        _ = explainer.compute(X_test, max_samples=max_samples)
        imp = explainer.feature_importance().head(top_k)

    fig_imp = px.bar(
        imp.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Importance globale des features (SHAP)",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Top features")
    st.dataframe(imp, use_container_width=True, hide_index=True)

elif view == "SHAP Local":
    st.markdown("### SHAP — Analyse locale (transaction)")
    st.markdown(
        '<div class="small">Explique une prédiction individuelle en affichant les contributions des features.</div>',
        unsafe_allow_html=True,
    )

    pool_n = st.sidebar.slider(
        "Taille pool",
        100,
        min(3000, len(test_df)),
        min(800, len(test_df)),
        step=100,
    )
    bg_size = st.sidebar.slider(
        "Taille background (SHAP)",
        100,
        min(5000, len(X_train)),
        min(1000, len(X_train)),
        step=100,
    )
    top_k = st.sidebar.slider(
        "Contributions affichées",
        5,
        min(40, len(feature_names)),
        min(15, len(feature_names)),
    )

    pred_filter = st.sidebar.selectbox("Filtre", ["Toutes", "Prédit fraude", "Prédit légitime"])

    pool = test_df.head(pool_n).copy()
    if pred_filter == "Prédit fraude":
        pool = pool[pool["prediction"] == 1]
    elif pred_filter == "Prédit légitime":
        pool = pool[pool["prediction"] == 0]

    if pool.empty:
        st.warning("Aucune transaction après filtrage.")
        st.stop()

    tx = st.selectbox("Transaction", pool["transaction_id"].tolist())
    idx = pool.index[pool["transaction_id"] == tx][0]
    local_pos = list(pool.index).index(idx)

    X_bg = X_train.sample(n=min(bg_size, len(X_train)), random_state=42)
    explainer = get_shap_explainer(model, X_bg, feature_names)

    with st.spinner("Calcul de l'explication locale SHAP…"):
        explainer.compute(X_test.loc[pool.index], max_samples=len(pool))
        local = explainer.local(local_pos)

    row = test_df.loc[idx]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transaction", tx)
    c2.metric("Probabilité fraude", f"{row['fraud_probability']:.1%}")
    c3.metric("Prédiction", "FRAUDE" if int(row["prediction"]) == 1 else "Légitime")
    c4.metric("Vérité", "FRAUDE" if int(row["actual"]) == 1 else "Légitime")

    shap_1d = local["shap_values"]
    order = np.argsort(np.abs(shap_1d))[::-1][:top_k]

    df_local = pd.DataFrame(
        {
            "Feature": [feature_names[i] for i in order],
            "SHAP": [float(shap_1d[i]) for i in order],
        }
    ).sort_values("SHAP", key=np.abs, ascending=True)

    fig = px.bar(df_local, x="SHAP", y="Feature", orientation="h", title="Contributions SHAP (local)")
    st.plotly_chart(fig, use_container_width=True)

elif view == "LIME":
    st.markdown("### LIME — Explication locale")
    st.markdown(
        '<div class="small">LIME approxime localement le modèle par un modèle simple (linéaire) pour expliquer une prédiction.</div>',
        unsafe_allow_html=True,
    )

    pool = test_df.head(200)
    tx = st.selectbox("Transaction", pool["transaction_id"].tolist())
    idx = pool.index[pool["transaction_id"] == tx][0]
    pos = list(X_test.index).index(idx) if idx in X_test.index else 0

    num_features = st.sidebar.slider(
        "Nb features LIME",
        5,
        min(40, len(feature_names)),
        min(12, len(feature_names)),
    )

    explainer = get_lime_explainer(model, X_train, feature_names)

    with st.spinner("Calcul LIME…"):
        exp = explainer.explain(X_test, pos, num_features=num_features)
        lime_df = explainer.to_df(exp, label=1)

    row = test_df.loc[idx]
    st.markdown(
        f"""
        <div class="card">
          <div><b>Transaction:</b> {tx}</div>
          <div><b>Prédiction:</b> {"FRAUDE" if int(row["prediction"])==1 else "Légitime"} — <b>p(fraude):</b> {row["fraud_probability"]:.1%}</div>
          <div><b>Vérité:</b> {"FRAUDE" if int(row["actual"])==1 else "Légitime"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lime_df["Direction"] = np.where(lime_df["Weight"] >= 0, "Vers fraude", "Vers légitime")
    fig = px.bar(
        lime_df.sort_values("Weight", key=np.abs, ascending=True),
        x="Weight",
        y="Feature",
        orientation="h",
        color="Direction",
        title="Contributions LIME",
    )
    st.plotly_chart(fig, use_container_width=True)

else:  # Comparaison
    st.markdown("### Comparaison SHAP vs LIME")
    st.markdown(
        '<div class="small">Compare les contributions des features calculées par SHAP et LIME pour une même transaction.</div>',
        unsafe_allow_html=True,
    )

    pool_n = st.sidebar.slider(
        "Taille pool",
        100,
        min(2000, len(test_df)),
        min(400, len(test_df)),
        step=100,
    )
    bg_size = st.sidebar.slider(
        "Taille background (SHAP)",
        100,
        min(5000, len(X_train)),
        min(1000, len(X_train)),
        step=100,
    )

    pool = test_df.head(pool_n)
    tx = st.selectbox("Transaction", pool["transaction_id"].tolist())
    idx = pool.index[pool["transaction_id"] == tx][0]
    local_pos = list(pool.index).index(idx)
    pos = list(X_test.index).index(idx) if idx in X_test.index else 0

    X_bg = X_train.sample(n=min(bg_size, len(X_train)), random_state=42)
    shap_exp = get_shap_explainer(model, X_bg, feature_names)
    lime_exp = get_lime_explainer(model, X_train, feature_names)

    with st.spinner("Calcul SHAP + LIME…"):
        shap_exp.compute(X_test.loc[pool.index], max_samples=len(pool))
        local = shap_exp.local(local_pos)
        lime_expl = lime_exp.explain(X_test, pos, num_features=min(25, len(feature_names)))

    comp_df, corr = compare_shap_lime(local["shap_values"], lime_expl, feature_names)

    row = test_df.loc[idx]
    c1, c2, c3 = st.columns(3)
    c1.metric("Transaction", tx)
    c2.metric("Probabilité", f"{row['fraud_probability']:.1%}")
    c3.metric("Corrélation |SHAP| vs |LIME|", "N/A" if np.isnan(corr) else f"{corr:.2f}")

    left, right = st.columns(2)
    with left:
        d = comp_df.sort_values("|SHAP|", ascending=False).head(20).sort_values("|SHAP|", ascending=True)
        st.plotly_chart(px.bar(d, x="SHAP", y="Feature", orientation="h", title="Top 20 SHAP (local)"), use_container_width=True)

    with right:
        d = comp_df.sort_values("|LIME|", ascending=False).head(20).sort_values("|LIME|", ascending=True)
        st.plotly_chart(px.bar(d, x="LIME", y="Feature", orientation="h", title="Top 20 LIME (local)"), use_container_width=True)

    st.markdown("#### Détails (Top 30)")
    st.dataframe(
        comp_df.sort_values("|SHAP|", ascending=False).head(30),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.caption("Projet — Financial Fraud Detection • Streamlit • SHAP • LIME")
