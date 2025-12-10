"""
Financial Fraud Detection Dashboard
====================================
Application Streamlit pour visualiser les données de fraude financière
et analyser les performances des modèles de détection.
"""

import streamlit as st

# Configuration de la page principale
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #333;
        margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #000;
        background-color: #2d2d2d;
        margin: 1rem 0;
    }
    .stMetric {
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<h1 class="main-header">Financial Fraud Detection</h1>', unsafe_allow_html=True)

# Description du projet
st.markdown("""
<div class="info-box">
<h3>A propos de ce projet</h3>
<p>Ce dashboard présente une analyse complète de la détection de fraudes financières utilisant 
des techniques de Machine Learning avancées et des méthodes d'explicabilité (XAI).</p>
</div>
""", unsafe_allow_html=True)

# Colonnes pour les statistiques principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Transactions",
        value="200K",
        delta="Dataset complet"
    )

with col2:
    st.metric(
        label="Features",
        value="40+",
        delta="Après feature engineering"
    )

with col3:
    st.metric(
        label="Modèles",
        value="2",
        delta="RF, XGB"
    )

with col4:
    st.metric(
        label="Meilleur F1",
        value="~70%",
        delta="Random Forest"
    )

st.markdown("---")

# Sections du dashboard
st.markdown("## Sections disponibles")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Data Exploration
    - Visualisation du dataset complet
    - Statistiques descriptives
    - Distribution des variables
    - Analyse des corrélations
    - Détection de la classe cible (fraude)
    
    **Allez dans la page "Data Exploration"**
    """)

with col2:
    st.markdown("""
    ### Model Results
    - Comparaison des 5 modèles
    - Métriques de performance
    - Matrices de confusion
    - Courbes ROC
    - Feature Importance
    
    **Allez dans la page "Model Results"**
    """)

with col3:
    st.markdown("""
    ### XAI Explainability
    - SHAP Values (global & local)
    - LIME Explanations
    - Interprétation des prédictions
    - Analyse de cas individuels
    
    **Allez dans la page "XAI Explainability"**
    """)

st.markdown("---")

# Sidebar
st.sidebar.markdown("## Navigation")
st.sidebar.info("""
Utilisez le menu ci-dessus pour naviguer entre les différentes sections :
- **Data Exploration** : Explorer le dataset
- **Model Results** : Voir les performances
- **XAI** : Comprendre les prédictions
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Projet")
st.sidebar.markdown("""
**Repository:** Financial-Fraud-Detection  
**Date:** Décembre 2025
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    <p>Projet École - Financial Fraud Detection avec Machine Learning</p>
    <p>Développé avec Streamlit • Python • Scikit-learn • XGBoost • LightGBM • SHAP</p>
</div>
""", unsafe_allow_html=True)
