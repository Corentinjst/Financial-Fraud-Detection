from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import os

try:
    from dotenv import load_dotenv
    load_dotenv()  
except ImportError:
    pass 

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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

# Configuration IA : récupérer automatiquement la clé depuis .env
api_key = os.getenv("OPENAI_API_KEY", "")

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
def get_shap_explainer(_model, _X_bg: pd.DataFrame, _feature_names: list[str], max_trees: int | None = None):
    return SHAPExplainer(_model, _X_bg, feature_names=_feature_names, max_trees=max_trees)

def get_lime_explainer(_model, _X_train: pd.DataFrame, _feature_names: list[str]):
    return LIMEExplainer(_model, _X_train, feature_names=_feature_names)


# ============================================================
# IA Explanation Helper
# ============================================================
def explain_with_ai(context: str, graph_data: dict, api_key: str) -> str:
    """
    Génère une explication en langage naturel d'un graphique XAI.
    
    Args:
        context: Le type de graphique (SHAP Global, SHAP Local, LIME, etc.)
        graph_data: Dictionnaire contenant les données du graphique
        api_key: Clé API OpenAI
    
    Returns:
        Explication textuelle générée par l'IA
    """
    try:
        client = OpenAI(api_key=api_key)
        
        # Construction du prompt selon le contexte
        prompt = f"""
        Vous êtes un expert en détection de fraude financière et en explainability (XAI).

        Contexte: {context}

        Données du graphique:
        {graph_data}

        Votre tâche:
        1. Expliquez de manière claire et accessible ce que montre ce graphique
        2. Identifiez les insights clés pour la détection de fraude
        3. Donnez des recommandations pratiques basées sur ces résultats
        4. Utilisez un langage accessible même pour des non-experts en ML

        Répondez en français de manière structurée et concise (200-300 mots).
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Vous êtes un expert en ML et détection de fraude, spécialisé dans l'explication de modèles (XAI)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Erreur lors de la génération de l'explication: {str(e)}"


# ============================================================
# Views
# ============================================================
if view == "SHAP Global":
    st.markdown("### SHAP — Analyse globale")
    st.markdown(
        '<div class="small">Les SHAP values mesurent la contribution moyenne de chaque feature à la prédiction.</div>',
        unsafe_allow_html=True,
    )

    # Détection Random Forest pour warning performance
    model_name = selected_model_file.lower()
    is_random_forest = 'random' in model_name or 'forest' in model_name or 'rf' in model_name
    
    # Avertissement pour Random Forest
    if is_random_forest:
        st.info("**Random Forest détecté** : les valeurs par défaut ont été réduites pour des performances optimales.")
    
    # Valeurs optimisées : pour RF, limiter encore plus agressivement
    max_samples = st.sidebar.slider(
        "Échantillons expliqués (SHAP)",
        10,
        50 if is_random_forest else min(500, len(X_test)),
        20 if is_random_forest else min(100, len(X_test)),
        step=5 if is_random_forest else 50,
    )
    bg_size = st.sidebar.slider(
        "Taille background (SHAP)",
        10,
        50 if is_random_forest else min(500, len(X_train)),
        20 if is_random_forest else min(100, len(X_train)),
        step=5 if is_random_forest else 50,
    )
    
    # Nouveau paramètre : limitation du nombre d'arbres pour RF
    max_trees = None
    if is_random_forest:
        max_trees = st.sidebar.slider(
            "Arbres RF utilisés (perf)",
            10,
            100,
            30,
            step=10,
            help="Limiter le nombre d'arbres accélère drastiquement SHAP pour Random Forest. 30 arbres donnent déjà de bonnes explications."
        )
    
    top_k = st.sidebar.slider(
        "Top features",
        5,
        min(40, len(feature_names)),
        min(20, len(feature_names)),
    )

    X_bg = X_train.sample(n=min(bg_size, len(X_train)), random_state=42)
    explainer = get_shap_explainer(model, X_bg, feature_names, max_trees=max_trees)

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
    
    # Bouton d'explication IA 
    if st.button("Expliquer ce graphique avec l'IA", key="ai_shap_global"):
        if not OPENAI_AVAILABLE:
            st.error("OpenAI non installé. Installez avec: `pip install openai`")
        elif not api_key:
            st.warning("Configurez votre clé API OpenAI dans le fichier .env")
            st.code("OPENAI_API_KEY=sk-votre-cle-api", language="bash")
        else:
            with st.spinner("Génération de l'explication IA..."):
                graph_data = {
                    "type": "SHAP Global - Importance des features",
                    "model": selected_model_file,
                    "top_features": imp.to_dict('records'),
                    "num_samples": max_samples,
                    "total_features": len(feature_names)
                }
                
                explanation = explain_with_ai(
                    "Analyse SHAP Globale - Importance des features pour la détection de fraude",
                    graph_data,
                    api_key
                )
                
                st.markdown("---")
                st.markdown("### Explication IA")
                st.markdown(explanation)

elif view == "SHAP Local":
    st.markdown("### SHAP — Analyse locale (transaction)")
    st.markdown(
        '<div class="small">Explique une prédiction individuelle en affichant les contributions des features.</div>',
        unsafe_allow_html=True,
    )

    # Valeurs optimisées pour performance
    model_name = selected_model_file.lower()
    is_random_forest = 'random' in model_name or 'forest' in model_name or 'rf' in model_name
    
    # Avertissement pour Random Forest
    if is_random_forest:
        st.info("⚡ **Random Forest détecté** : paramètres optimisés pour la performance.")
    
    pool_n = st.sidebar.slider(
        "Taille pool",
        10,
        50 if is_random_forest else min(500, len(test_df)),
        20 if is_random_forest else min(200, len(test_df)),
        step=5 if is_random_forest else 50,
    )
    bg_size = st.sidebar.slider(
        "Taille background (SHAP)",
        10,
        50 if is_random_forest else min(500, len(X_train)),
        20 if is_random_forest else min(100, len(X_train)),
        step=5 if is_random_forest else 50,
    )
    
    # Nouveau paramètre : limitation du nombre d'arbres pour RF
    max_trees = None
    if is_random_forest:
        max_trees = st.sidebar.slider(
            "Arbres RF utilisés (perf)",
            10,
            100,
            30,
            step=10,
            help="Limiter le nombre d'arbres accélère drastiquement SHAP. 30 arbres suffisent pour de bonnes explications."
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
    explainer = get_shap_explainer(model, X_bg, feature_names, max_trees=max_trees)

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
    
    # Bouton d'explication IA 
    if st.button("Expliquer cette transaction avec l'IA", key="ai_shap_local"):
        if not OPENAI_AVAILABLE:
            st.error("OpenAI non installé. Installez avec: `pip install openai`")
        elif not api_key:
            st.warning("Configurez votre clé API OpenAI dans le fichier .env")
            st.code("OPENAI_API_KEY=sk-votre-cle-api", language="bash")
        else:
            with st.spinner("Analyse de la transaction..."):
                graph_data = {
                    "type": "SHAP Local - Explication d'une transaction",
                    "transaction_id": tx,
                    "fraud_probability": float(row['fraud_probability']),
                    "prediction": "FRAUDE" if int(row["prediction"]) == 1 else "Légitime",
                    "actual": "FRAUDE" if int(row["actual"]) == 1 else "Légitime",
                    "top_contributions": df_local.to_dict('records'),
                    "model": selected_model_file
                }
                
                explanation = explain_with_ai(
                    "Analyse SHAP Locale - Explication des contributions pour une transaction spécifique",
                    graph_data,
                    api_key
                )
                
                st.markdown("---")
                st.markdown("### Explication IA de la transaction")
                st.markdown(explanation)

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
    
    # Bouton d'explication IA 
    if st.button("Expliquer cette prédiction LIME avec l'IA", key="ai_lime"):
        if not OPENAI_AVAILABLE:
            st.error("OpenAI non installé. Installez avec: `pip install openai`")
        elif not api_key:
            st.warning("Configurez votre clé API OpenAI dans le fichier .env")
            st.code("OPENAI_API_KEY=sk-votre-cle-api", language="bash")
        else:
            with st.spinner("Analyse LIME en cours..."):
                graph_data = {
                    "type": "LIME - Explication locale linéaire",
                    "transaction_id": tx,
                    "fraud_probability": float(row['fraud_probability']),
                    "prediction": "FRAUDE" if int(row["prediction"]) == 1 else "Légitime",
                    "actual": "FRAUDE" if int(row["actual"]) == 1 else "Légitime",
                    "lime_contributions": lime_df.to_dict('records'),
                    "model": selected_model_file,
                    "num_features": num_features
                }
                
                explanation = explain_with_ai(
                    "Analyse LIME - Approximation linéaire locale du modèle",
                    graph_data,
                    api_key
                )
                
                st.markdown("---")
                st.markdown("### Explication IA (LIME)")
                st.markdown(explanation)

else:  # Comparaison
    st.markdown("### Comparaison SHAP vs LIME")
    st.markdown(
        '<div class="small">Compare les contributions des features calculées par SHAP et LIME pour une même transaction.</div>',
        unsafe_allow_html=True,
    )

    # Valeurs optimisées pour performance
    model_name = selected_model_file.lower()
    is_random_forest = 'random' in model_name or 'forest' in model_name or 'rf' in model_name
    
    # Avertissement pour Random Forest
    if is_random_forest:
        st.warning("⚠️ **Random Forest détecté** : la comparaison SHAP+LIME peut être très lente (30-60s). Considérez utiliser les vues séparées.")
    
    pool_n = st.sidebar.slider(
        "Taille pool",
        10,
        30 if is_random_forest else min(500, len(test_df)),
        15 if is_random_forest else min(200, len(test_df)),
        step=5 if is_random_forest else 50,
    )
    bg_size = st.sidebar.slider(
        "Taille background (SHAP)",
        10,
        50 if is_random_forest else min(500, len(X_train)),
        20 if is_random_forest else min(100, len(X_train)),
        step=5 if is_random_forest else 50,
    )
    
    # Nouveau paramètre : limitation du nombre d'arbres pour RF
    max_trees = None
    if is_random_forest:
        max_trees = st.sidebar.slider(
            "Arbres RF utilisés (perf)",
            10,
            100,
            25,
            step=5,
            help="Limiter drastiquement pour la comparaison SHAP+LIME."
        )

    pool = test_df.head(pool_n)
    tx = st.selectbox("Transaction", pool["transaction_id"].tolist())
    idx = pool.index[pool["transaction_id"] == tx][0]
    local_pos = list(pool.index).index(idx)
    pos = list(X_test.index).index(idx) if idx in X_test.index else 0

    X_bg = X_train.sample(n=min(bg_size, len(X_train)), random_state=42)
    shap_exp = get_shap_explainer(model, X_bg, feature_names, max_trees=max_trees)
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
    
    # Bouton d'explication IA 
    if st.button("Expliquer la comparaison SHAP vs LIME avec l'IA", key="ai_comparison"):
        if not OPENAI_AVAILABLE:
            st.error("OpenAI non installé. Installez avec: `pip install openai`")
        elif not api_key:
            st.warning("Configurez votre clé API OpenAI dans le fichier .env")
            st.code("OPENAI_API_KEY=sk-votre-cle-api", language="bash")
        else:
            with st.spinner("Analyse comparative en cours..."):
                graph_data = {
                    "type": "Comparaison SHAP vs LIME",
                    "transaction_id": tx,
                    "fraud_probability": float(row['fraud_probability']),
                    "prediction": "FRAUDE" if int(row["prediction"]) == 1 else "Légitime",
                    "actual": "FRAUDE" if int(row["actual"]) == 1 else "Légitime",
                    "correlation": "N/A" if np.isnan(corr) else float(corr),
                    "top_shap_features": comp_df.sort_values("|SHAP|", ascending=False).head(10).to_dict('records'),
                    "top_lime_features": comp_df.sort_values("|LIME|", ascending=False).head(10).to_dict('records'),
                    "model": selected_model_file
                }
                
                explanation = explain_with_ai(
                    "Comparaison SHAP vs LIME - Convergence des méthodes d'explainability",
                    graph_data,
                    api_key
                )
                
                st.markdown("---")
                st.markdown("### Explication IA de la comparaison")
                st.markdown(explanation)

st.markdown("---")
st.caption("Projet — Financial Fraud Detection • Streamlit • SHAP • LIME")
