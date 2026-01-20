"""
Data Exploration
================
Exploration et visualisation du dataset de transactions financières.
Compatible avec : data/train.parquet, data/val.parquet, data/test.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# Page config + style léger
# ============================================================
st.set_page_config(
    page_title="Data Exploration - Fraud Detection",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem; }
      .sub-title { color: #6b7280; margin-top: 0; margin-bottom: 1.2rem; }
      .small { color: #6b7280; font-size: 0.9rem; }
      .card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.02);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Data Exploration</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Exploration et visualisation des transactions financières (splits train/val/test).</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # repo/
DATA_DIR = ROOT / "data"

# ============================================================
# Data loader
# ============================================================
@st.cache_data
def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame | None, str | None]:
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        return None, f"Fichier introuvable: {path}"
    df = pd.read_parquet(path)
    return df, None


# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Paramètres")

split = st.sidebar.selectbox("Jeu de données", ["train", "val", "test"], index=0)

df, err = load_split(DATA_DIR, split)
if err:
    st.error("Impossible de charger les données.")
    st.info("Attendu dans /data : train.parquet, val.parquet, test.parquet")
    st.code(err)
    st.stop()

if df is None or df.empty:
    st.error("Le dataset est vide.")
    st.stop()

# ------------------------------------------------------------
# Target selection (fraude)
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Cible (fraude)")

all_cols = list(df.columns)
binary_cols = [c for c in all_cols if df[c].dropna().nunique() == 2]

target_col = st.sidebar.selectbox(
    "Colonne cible",
    options=(binary_cols + [c for c in all_cols if c not in binary_cols]) if all_cols else [],
    index=0 if binary_cols else 0,
)

# ------------------------------------------------------------
# Filters
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")

df_filtered = df.copy()

# filtre catégorie (choix colonne)
cat_cols = df_filtered.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
cat_choice = st.sidebar.selectbox("Filtre catégorie (optionnel)", ["Aucun"] + cat_cols, index=0)

if cat_choice != "Aucun":
    values = df_filtered[cat_choice].dropna().astype(str).unique().tolist()
    values = sorted(values)
    selected_value = st.sidebar.selectbox(f"Valeur de {cat_choice}", ["Tous"] + values, index=0)
    if selected_value != "Tous":
        df_filtered = df_filtered[df_filtered[cat_choice].astype(str) == selected_value]

# filtre numérique (choix colonne)
num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
num_choice = st.sidebar.selectbox("Filtre numérique (optionnel)", ["Aucun"] + num_cols, index=0)

if num_choice != "Aucun":
    vmin = float(df[num_choice].min())
    vmax = float(df[num_choice].max())
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
        rng = st.sidebar.slider(
            f"Plage de {num_choice}",
            min_value=vmin,
            max_value=vmax,
            value=(vmin, vmax),
        )
        df_filtered = df_filtered[(df_filtered[num_choice] >= rng[0]) & (df_filtered[num_choice] <= rng[1])]

# échantillon
st.sidebar.markdown("---")
sample_size = st.sidebar.slider("Échantillon affiché", 100, min(20000, len(df_filtered)), min(2000, len(df_filtered)), step=100)

# ============================================================
# Overview
# ============================================================
st.markdown("## Overview du dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Split", split)
with col2:
    st.metric("Lignes", f"{len(df):,}")
with col3:
    st.metric("Colonnes", f"{df.shape[1]}")
with col4:
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    st.metric("Mémoire", f"{mem_mb:.1f} MB")

# taux de fraude si cible binaire exploitable
if target_col in df.columns and df[target_col].dropna().nunique() == 2:
    try:
        fraud_rate = float((df[target_col] == 1).mean())
        st.markdown(
            f"""
            <div class="card">
              <b>Taux de fraude (sur {split}):</b> {fraud_rate:.2%}
              <div class="small">Calculé comme la proportion de la classe 1 dans la colonne cible sélectionnée.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

st.markdown("### Aperçu (après filtres)")
st.dataframe(df_filtered.head(sample_size), use_container_width=True)

with st.expander("Informations sur les colonnes"):
    info = pd.DataFrame(
        {
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count(),
            "Null": df.isnull().sum(),
            "Uniques": df.nunique(dropna=True),
        }
    )
    st.dataframe(info, use_container_width=True)

with st.expander("Statistiques descriptives (numériques)"):
    st.dataframe(df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)

st.markdown("---")

# ============================================================
# Target distribution
# ============================================================
st.markdown("## Distribution de la cible")

if target_col in df.columns and df[target_col].dropna().nunique() == 2:
    counts = df[target_col].value_counts().sort_index()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    total = int(n0 + n1)
    rate = (n1 / total) if total else 0.0
    imbalance = (n0 / n1) if n1 else np.inf

    c1, c2 = st.columns(2)

    with c1:
        fig_pie = px.pie(
            values=[n0, n1],
            names=["Non-Fraude (0)", "Fraude (1)"],
            title="Répartition Fraude / Non-Fraude",
            hole=0.45,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_bar = px.bar(
            x=["Non-Fraude (0)", "Fraude (1)"],
            y=[n0, n1],
            title="Nombre de transactions par classe",
            text=[f"{n0:,}", f"{n1:,}"],
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.info(
        f"""
**Déséquilibre des classes**
- Légitimes (0): **{n0:,}** ({(1-rate):.2%})
- Fraudes (1): **{n1:,}** ({rate:.2%})
- Ratio d’imbalance: **1:{int(imbalance) if np.isfinite(imbalance) else '∞'}**
"""
    )
else:
    st.warning("La colonne cible sélectionnée n'est pas binaire (0/1). Choisis une autre colonne cible.")

st.markdown("---")

# ============================================================
# Variable distribution (numeric)
# ============================================================
st.markdown("## Distribution des variables")

numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.warning("Aucune colonne numérique détectée.")
else:
    left, right = st.columns([1, 3])

    with left:
        selected_var = st.selectbox("Variable", numeric_cols, index=0)
        chart_type = st.radio("Graphique", ["Histogramme", "Boxplot"], index=0)
        color_by_target = st.checkbox("Colorer par cible", value=True)

    with right:
        sample_df = df_filtered.sample(min(sample_size, len(df_filtered))) if len(df_filtered) > sample_size else df_filtered

        if chart_type == "Histogramme":
            if color_by_target and target_col in df.columns and df[target_col].dropna().nunique() == 2:
                fig = px.histogram(sample_df, x=selected_var, color=target_col, barmode="overlay", opacity=0.6,
                                   title=f"Distribution de {selected_var} (coloré par {target_col})")
            else:
                fig = px.histogram(sample_df, x=selected_var, title=f"Distribution de {selected_var}")
        else:
            if color_by_target and target_col in df.columns and df[target_col].dropna().nunique() == 2:
                fig = px.box(sample_df, x=target_col, y=selected_var, title=f"Boxplot de {selected_var} (par {target_col})")
            else:
                fig = px.box(sample_df, y=selected_var, title=f"Boxplot de {selected_var}")

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================
# Categorical analysis (optional)
# ============================================================
st.markdown("## Analyse catégorielle")

if len(cat_cols) == 0:
    st.warning("Aucune colonne catégorielle détectée.")
else:
    c1, c2 = st.columns([1, 3])
    with c1:
        cat_var = st.selectbox("Variable catégorielle", cat_cols, index=0)
        topn = st.slider("Top catégories", 5, 30, 10)
        show_target_rate = st.checkbox("Afficher taux de fraude par catégorie", value=True)

    with c2:
        counts = df[cat_var].astype(str).value_counts().head(topn)
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            title=f"Top {topn} catégories — {cat_var}",
            text=[f"{v:,}" for v in counts.values],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=cat_var, yaxis_title="Nombre")
        st.plotly_chart(fig, use_container_width=True)

    if show_target_rate and target_col in df.columns and df[target_col].dropna().nunique() == 2:
        tmp = df[[cat_var, target_col]].dropna().copy()
        tmp[cat_var] = tmp[cat_var].astype(str)
        rate_df = tmp.groupby(cat_var)[target_col].mean().sort_values(ascending=False).head(topn).reset_index()
        rate_df[target_col] = rate_df[target_col] * 100

        fig_rate = px.bar(
            rate_df,
            x=cat_var,
            y=target_col,
            title=f"Taux de fraude (%) par {cat_var} — Top {topn}",
            text=rate_df[target_col].round(2),
        )
        fig_rate.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_rate.update_layout(yaxis_title="Taux (%)", xaxis_title=cat_var)
        st.plotly_chart(fig_rate, use_container_width=True)

st.markdown("---")

# ============================================================
# Correlation matrix
# ============================================================
st.markdown("## Corrélations")

numeric_cols_full = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols_full) < 2:
    st.warning("Pas assez de colonnes numériques pour calculer une corrélation.")
else:
    max_features = st.slider("Nb features (corrélation)", 5, min(30, len(numeric_cols_full)), 15)

    if target_col in df.columns and target_col in numeric_cols_full and df[target_col].dropna().nunique() == 2:
        corr_to_target = df[numeric_cols_full].corr(method="pearson")[target_col].abs().sort_values(ascending=False)
        top_features = corr_to_target.head(max_features).index.tolist()
    else:
        top_features = numeric_cols_full[:max_features]

    corr_matrix = df[top_features].corr(method="pearson")

    fig_corr = px.imshow(
        corr_matrix,
        title="Matrice de corrélation (Pearson)",
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    if target_col in df.columns and target_col in numeric_cols_full and df[target_col].dropna().nunique() == 2:
        top_corr_df = pd.DataFrame({"Feature": corr_to_target.head(15).index, "Corrélation (abs)": corr_to_target.head(15).values})
        fig_top = px.bar(
            top_corr_df.sort_values("Corrélation (abs)", ascending=True),
            x="Corrélation (abs)",
            y="Feature",
            orientation="h",
            title=f"Top corrélations avec {target_col}",
        )
        fig_top.update_layout(height=520)
        st.plotly_chart(fig_top, use_container_width=True)

st.markdown("---")

# ============================================================
# Scatter plot
# ============================================================
st.markdown("## Scatter plot")

if len(num_cols) >= 2:
    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox("Variable X", num_cols, index=0)
    with c2:
        y_var = st.selectbox("Variable Y", num_cols, index=1 if len(num_cols) > 1 else 0)
    with c3:
        color_target = st.checkbox("Colorer par cible", value=True)

    sample_df = df_filtered.sample(min(5000, len(df_filtered)))

    if color_target and target_col in df.columns and df[target_col].dropna().nunique() == 2:
        fig_scatter = px.scatter(sample_df, x=x_var, y=y_var, color=target_col, opacity=0.6, title=f"{x_var} vs {y_var}")
    else:
        fig_scatter = px.scatter(sample_df, x=x_var, y=y_var, opacity=0.6, title=f"{x_var} vs {y_var}")

    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning("Pas assez de colonnes numériques pour un scatter plot.")

st.markdown("---")

# ============================================================
# Downloads
# ============================================================
st.markdown("## Téléchargements")

col1, col2 = st.columns(2)

with col1:
    csv = df_filtered.head(10000).to_csv(index=False)
    st.download_button(
        label="Télécharger un échantillon filtré (CSV)",
        data=csv,
        file_name=f"{split}_sample.csv",
        mime="text/csv",
    )

with col2:
    num_desc = df.select_dtypes(include=[np.number]).describe().T
    st.download_button(
        label="Télécharger statistiques numériques (CSV)",
        data=num_desc.to_csv(),
        file_name=f"{split}_numeric_stats.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Projet — Financial Fraud Detection • Data Exploration • Streamlit")
