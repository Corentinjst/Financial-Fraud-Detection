"""
XAI Explainability
==================
Page d'explicabilité des modèles avec SHAP et LIME.
Utilise les vraies données et modèles entraînés.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Ajouter le dossier parent au path pour importer xai_utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def normalize_shap_values(shap_values):
    """
    Normalize SHAP values to ensure they are a 2D numpy array.
    Handles SHAP Explanation objects and 3D arrays (binary classification).
    """
    # Handle SHAP Explanation object
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
    
    # Ensure numpy array
    shap_values = np.array(shap_values)
    
    # Handle 3D array (samples x features x classes) for binary classification
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # Take positive class
    
    return shap_values

# Configuration de la page
st.set_page_config(
    page_title="XAI Explainability - Fraud Detection",
    page_icon="",
    layout="wide"
)

# Titre
st.markdown("# XAI - Explainability")
st.markdown("Comprendre les décisions des modèles grace a SHAP et LIME.")

# =========================================
# Chargement des données et modèles
# =========================================
@st.cache_data
def load_data():
    """Charge les données depuis les fichiers parquet."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_path, "data")
    
    try:
        X_test = pd.read_parquet(os.path.join(data_path, "X_test.parquet"))
        y_test = pd.read_parquet(os.path.join(data_path, "y_test.parquet"))
        X_train = pd.read_parquet(os.path.join(data_path, "X_train.parquet"))
        return X_train, X_test, y_test, True
    except FileNotFoundError as e:
        st.warning(f"Fichiers de données non trouvés: {e}")
        return None, None, None, False

@st.cache_resource
def load_model(model_name):
    """Charge un modèle sauvegardé."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_path = os.path.join(base_path, "models")
    
    model_files = {
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }
    
    try:
        model_file = os.path.join(models_path, model_files.get(model_name, ""))
        if os.path.exists(model_file):
            return joblib.load(model_file), True
        else:
            return None, False
    except Exception as e:
        st.warning(f"Erreur lors du chargement du modèle: {e}")
        return None, False

@st.cache_resource
def get_shap_explainer(_model, _X_background, feature_names):
    """Crée et cache l'explainer SHAP."""
    from xai_utils import SHAPExplainer
    return SHAPExplainer(_model, _X_background, feature_names, model_type='tree')

@st.cache_resource
def get_lime_explainer(_model, _X_train, feature_names):
    """Crée et cache l'explainer LIME."""
    from xai_utils import LIMEExplainer
    return LIMEExplainer(_model, _X_train, feature_names)

# Charger les données
X_train, X_test, y_test, data_loaded = load_data()

# Sidebar
st.sidebar.markdown("## Options")

model_choice = st.sidebar.selectbox(
    "Modele",
    ["Random Forest", "XGBoost"]
)

xai_method = st.sidebar.radio(
    "Méthode d'explicabilité",
    ["SHAP Global", "SHAP Local", "LIME", "Comparaison"]
)

# Vérifier si les données sont chargées
if not data_loaded:
    st.error("Les données n'ont pas pu etre chargées. Vérifiez que les fichiers existent dans le dossier data/")
    st.info("Fichiers attendus: X_train.parquet, X_test.parquet, y_test.parquet")
    st.stop()

# Charger le modèle
model, model_loaded = load_model(model_choice)

if not model_loaded:
    st.error(f"Le modele '{model_choice}' n'a pas pu etre chargé.")
    st.info("Assurez-vous d'avoir sauvegardé les modèles dans le dossier models/ avec joblib.")
    st.code("""
# Dans votre notebook, sauvegardez les modèles ainsi:
import joblib
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(xgb_model, 'models/xgboost.pkl')
joblib.dump(lgb_model, 'models/lightgbm.pkl')
    """)
    st.stop()

# Feature names
feature_names = list(X_test.columns)

# Créer les dataframes avec prédictions
@st.cache_data
def prepare_test_data(_model, X_test_df, y_test_df):
    """Prépare les données de test avec prédictions."""
    predictions = _model.predict(X_test_df)
    probas = _model.predict_proba(X_test_df)[:, 1]
    
    test_df = X_test_df.copy()
    test_df['prediction'] = predictions
    test_df['fraud_probability'] = probas
    test_df['actual'] = y_test_df.values.ravel() if hasattr(y_test_df, 'values') else y_test_df
    test_df['transaction_id'] = [f'TXN_{i:05d}' for i in range(len(test_df))]
    
    return test_df

test_data = prepare_test_data(model, X_test, y_test)

# =========================================
# Vue 1: SHAP Global
# =========================================
if xai_method == "SHAP Global":
    st.markdown("---")
    st.markdown("## SHAP - Analyse Globale")
    st.markdown("""
    Les **SHAP values** (SHapley Additive exPlanations) mesurent la contribution de chaque 
    feature a la prédiction du modele. Une valeur SHAP positive pousse la prédiction vers 
    la classe "Fraude", une valeur négative vers "Non-Fraude".
    """)
    
    # Paramètres
    max_samples = st.sidebar.slider("Nombre d'échantillons pour SHAP", 100, 1000, 500)
    
    with st.spinner("Calcul des SHAP values en cours..."):
        # Créer l'explainer SHAP
        shap_explainer = get_shap_explainer(model, X_train.sample(min(1000, len(X_train)), random_state=42), feature_names)
        
        # Calculer les SHAP values
        shap_values = shap_explainer.compute_shap_values(X_test, max_samples=max_samples)
        shap_values = normalize_shap_values(shap_values)
        
        # Feature importance
        importance_df = shap_explainer.get_feature_importance()
    
    # Summary Plot (Bar)
    st.markdown("### Importance globale des features")
    
    fig_bar = px.bar(
        importance_df.sort_values('Importance', ascending=True),
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'SHAP Feature Importance - {model_choice}',
        color='Importance',
        color_continuous_scale='Reds'
    )
    fig_bar.update_layout(height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Beeswarm plot
    st.markdown("### Distribution des SHAP Values (Beeswarm)")
    
    # Créer un scatter plot simulant le beeswarm
    beeswarm_data = []
    X_explained = shap_explainer.X_explained
    
    for i, feature in enumerate(feature_names):
        for j in range(min(200, len(shap_values))):
            # Normaliser les valeurs des features pour la coloration
            feat_values = X_explained[feature].values
            feat_min, feat_max = feat_values.min(), feat_values.max()
            if feat_max > feat_min:
                normalized_value = (X_explained.iloc[j][feature] - feat_min) / (feat_max - feat_min)
            else:
                normalized_value = 0.5
            
            beeswarm_data.append({
                'Feature': feature,
                'SHAP Value': shap_values[j, i],
                'Feature Value': normalized_value
            })
    
    beeswarm_df = pd.DataFrame(beeswarm_data)
    
    fig_beeswarm = px.scatter(
        beeswarm_df,
        x='SHAP Value',
        y='Feature',
        color='Feature Value',
        color_continuous_scale='RdBu_r',
        title='Distribution des SHAP Values par Feature'
    )
    fig_beeswarm.update_traces(marker=dict(size=4, opacity=0.6))
    fig_beeswarm.update_layout(height=600)
    st.plotly_chart(fig_beeswarm, use_container_width=True)
    
    # Dependence Plot
    st.markdown("### Dependence Plot")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_feature = st.selectbox(
            "Feature a analyser",
            feature_names,
            index=0
        )
        interaction_feature = st.selectbox(
            "Feature d'interaction",
            ['Aucune'] + feature_names,
            index=0
        )
    
    with col2:
        feature_idx = feature_names.index(selected_feature)
        
        dep_df = pd.DataFrame({
            selected_feature: X_explained[selected_feature].values[:len(shap_values)],
            'SHAP Value': shap_values[:, feature_idx]
        })
        
        if interaction_feature != 'Aucune':
            int_idx = feature_names.index(interaction_feature)
            dep_df['Interaction'] = X_explained[interaction_feature].values[:len(shap_values)]
            
            fig_dep = px.scatter(
                dep_df,
                x=selected_feature,
                y='SHAP Value',
                color='Interaction',
                color_continuous_scale='RdBu_r',
                title=f'Dependence Plot: {selected_feature} (coloré par {interaction_feature})',
                opacity=0.6
            )
        else:
            fig_dep = px.scatter(
                dep_df,
                x=selected_feature,
                y='SHAP Value',
                title=f'Dependence Plot: {selected_feature}',
                color_discrete_sequence=['#667eea'],
                opacity=0.6
            )
        
        fig_dep.update_layout(height=400)
        st.plotly_chart(fig_dep, use_container_width=True)

# =========================================
# Vue 2: SHAP Local
# =========================================
elif xai_method == "SHAP Local":
    st.markdown("---")
    st.markdown("## SHAP - Analyse Locale")
    st.markdown("""
    L'analyse locale explique une **prédiction individuelle**. 
    Sélectionnez une transaction pour comprendre pourquoi le modele l'a classée comme fraude ou non.
    """)
    
    # Calculer SHAP values si pas déjà fait
    max_samples = st.sidebar.slider("Nombre d'échantillons", 100, 500, 200)
    
    with st.spinner("Calcul des SHAP values..."):
        shap_explainer = get_shap_explainer(model, X_train.sample(min(1000, len(X_train)), random_state=42), feature_names)
        shap_values = shap_explainer.compute_shap_values(X_test, max_samples=max_samples)
        shap_values = normalize_shap_values(shap_values)
    
    # Sélection de la transaction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Sélectionner une transaction")
        
        # Filtrer par type de prédiction
        pred_filter = st.radio(
            "Filtrer par prédiction",
            ["Toutes", "Fraudes détectées", "Non-fraudes"]
        )
        
        # Filtrer les données
        filtered_data = test_data.head(max_samples).copy()
        
        if pred_filter == "Fraudes détectées":
            filtered_data = filtered_data[filtered_data['prediction'] == 1]
        elif pred_filter == "Non-fraudes":
            filtered_data = filtered_data[filtered_data['prediction'] == 0]
        
        if len(filtered_data) == 0:
            st.warning("Aucune transaction ne correspond au filtre.")
            st.stop()
        
        selected_txn = st.selectbox(
            "Transaction",
            filtered_data['transaction_id'].tolist()
        )
        
        # Trouver l'index
        txn_idx = filtered_data[filtered_data['transaction_id'] == selected_txn].index[0]
        txn_idx_in_shap = list(test_data.head(max_samples).index).index(txn_idx)
        txn_data = test_data.loc[txn_idx]
    
    with col2:
        st.markdown("### Détails de la transaction")
        
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            amount_val = txn_data.get('amount', txn_data.get('Amount', 0))
            st.metric("Montant", f"${amount_val:,.2f}")
            pred_label = "FRAUDE" if txn_data['prediction'] == 1 else "Légitime"
            st.metric("Prédiction", pred_label)
        
        with details_col2:
            st.metric("Probabilité de fraude", f"{txn_data['fraud_probability']:.1%}")
            actual_label = "FRAUDE" if txn_data['actual'] == 1 else "Légitime"
            st.metric("Réalité", actual_label)
    
    st.markdown("---")
    
    # Obtenir l'explication SHAP pour cette transaction
    explanation = shap_explainer.get_sample_explanation(txn_idx_in_shap)
    
    # Waterfall Plot
    st.markdown("### Waterfall Plot - Contribution des features")
    
    # Trier par valeur absolue
    shap_vals = explanation['shap_values']
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
    
    features_sorted = [feature_names[i] for i in sorted_indices]
    values_sorted = [shap_vals[i] for i in sorted_indices]
    
    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(features_sorted) + ["total"],
        x=features_sorted + ["Prédiction finale"],
        y=values_sorted + [sum(values_sorted) + explanation['base_value']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#2ecc71"}},
        increasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#667eea"}}
    ))
    
    fig_waterfall.update_layout(
        title=f"Explication SHAP - {selected_txn}",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Force Plot (horizontal bar)
    st.markdown("### Force Plot")
    
    force_df = pd.DataFrame({
        'Feature': features_sorted,
        'Contribution': values_sorted,
        'Direction': ['Vers Fraude' if v > 0 else 'Vers Non-Fraude' for v in values_sorted]
    })
    
    fig_force = px.bar(
        force_df,
        x='Contribution',
        y='Feature',
        orientation='h',
        color='Direction',
        color_discrete_map={'Vers Fraude': '#e74c3c', 'Vers Non-Fraude': '#2ecc71'},
        title='Contributions des features a la prédiction'
    )
    fig_force.update_layout(height=400)
    st.plotly_chart(fig_force, use_container_width=True)

# =========================================
# Vue 3: LIME
# =========================================
elif xai_method == "LIME":
    st.markdown("---")
    st.markdown("## LIME - Local Interpretable Model-agnostic Explanations")
    st.markdown("""
    **LIME** crée un modèle simple (linéaire) autour d'une prédiction spécifique pour expliquer 
    le comportement local du modèle complexe.
    """)
    
    # Créer l'explainer LIME
    with st.spinner("Initialisation de LIME..."):
        lime_explainer = get_lime_explainer(model, X_train.sample(min(5000, len(X_train)), random_state=42), feature_names)
    
    # Sélection de la transaction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Sélectionner une transaction")
        
        show_fraud = st.checkbox("Afficher uniquement les fraudes", value=False)
        
        if show_fraud:
            filtered_df = test_data[test_data['prediction'] == 1].head(100)
        else:
            filtered_df = test_data.head(100)
        
        if len(filtered_df) == 0:
            st.warning("Aucune transaction ne correspond au filtre.")
            st.stop()
        
        selected_txn_lime = st.selectbox(
            "Transaction a expliquer",
            filtered_df['transaction_id'].tolist(),
            key="lime_txn"
        )
        
        txn_idx_lime = filtered_df[filtered_df['transaction_id'] == selected_txn_lime].index[0]
        txn_data_lime = test_data.loc[txn_idx_lime]
    
    with col2:
        st.markdown("### Explication LIME")
        
        pred_label = "FRAUDE" if txn_data_lime['prediction'] == 1 else "Légitime"
        prob = txn_data_lime['fraud_probability']
        
        st.markdown(f"""
        **Transaction:** {selected_txn_lime}  
        **Prédiction:** {pred_label}  
        **Probabilité de fraude:** {prob:.1%}
        """)
    
    # Calculer l'explication LIME
    num_features = st.sidebar.slider("Nombre de features a afficher", 5, len(feature_names), 10)
    
    with st.spinner("Calcul de l'explication LIME..."):
        # Trouver l'index dans X_test
        idx_in_xtest = list(X_test.index).index(txn_idx_lime) if txn_idx_lime in X_test.index else 0
        lime_explanation = lime_explainer.explain_instance(X_test, idx_in_xtest, num_features=num_features)
        lime_df = lime_explainer.get_explanation_df(lime_explanation, label=1)
    
    # Graphique des contributions LIME
    lime_df['Color'] = lime_df['Weight'].apply(lambda x: 'Vers Fraude' if x > 0 else 'Vers Non-Fraude')
    lime_df = lime_df.sort_values('Weight', key=abs, ascending=True)
    
    fig_lime = px.bar(
        lime_df,
        x='Weight',
        y='Feature',
        orientation='h',
        color='Color',
        color_discrete_map={'Vers Fraude': '#e74c3c', 'Vers Non-Fraude': '#2ecc71'},
        title=f'Explication LIME - {selected_txn_lime}'
    )
    fig_lime.update_layout(height=500)
    st.plotly_chart(fig_lime, use_container_width=True)
    
    # Tableau des valeurs
    st.markdown("### Valeurs des features pour cette transaction")
    
    feature_values_data = []
    for feat in feature_names[:10]:
        val = txn_data_lime.get(feat, 'N/A')
        lime_weight = lime_df[lime_df['Feature'].str.contains(feat, na=False)]['Weight'].values
        weight = lime_weight[0] if len(lime_weight) > 0 else 0
        
        feature_values_data.append({
            'Feature': feat,
            'Valeur': f"{val:.4f}" if isinstance(val, (int, float)) else val,
            'Contribution LIME': f"{weight:.4f}",
            'Impact': 'Vers Fraude' if weight > 0 else 'Vers Légitime'
        })
    
    feature_values_df = pd.DataFrame(feature_values_data)
    st.dataframe(feature_values_df, use_container_width=True, hide_index=True)

# =========================================
# Vue 4: Comparaison SHAP vs LIME
# =========================================
else:  # Comparaison
    st.markdown("---")
    st.markdown("## Comparaison SHAP vs LIME")
    st.markdown("Comparez les deux méthodes d'explicabilité pour la meme transaction.")
    
    # Initialiser les explainers
    with st.spinner("Initialisation des explainers..."):
        shap_explainer = get_shap_explainer(model, X_train.sample(min(1000, len(X_train)), random_state=42), feature_names)
        lime_explainer = get_lime_explainer(model, X_train.sample(min(5000, len(X_train)), random_state=42), feature_names)
        
        # Calculer SHAP values
        max_samples = 200
        shap_values = shap_explainer.compute_shap_values(X_test, max_samples=max_samples)
        shap_values = normalize_shap_values(shap_values)
    
    # Sélection de la transaction
    selected_txn_comp = st.selectbox(
        "Transaction a comparer",
        test_data.head(max_samples)['transaction_id'].tolist(),
        key="comp_txn"
    )
    
    txn_idx_comp = test_data[test_data['transaction_id'] == selected_txn_comp].index[0]
    txn_idx_in_shap = list(test_data.head(max_samples).index).index(txn_idx_comp)
    txn_data_comp = test_data.loc[txn_idx_comp]
    
    # Infos transaction
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transaction", selected_txn_comp)
    with col2:
        st.metric("Prédiction", "FRAUDE" if txn_data_comp['prediction'] == 1 else "Légitime")
    with col3:
        st.metric("Probabilité", f"{txn_data_comp['fraud_probability']:.1%}")
    
    st.markdown("---")
    
    # Calculer les explications
    with st.spinner("Calcul des explications..."):
        # SHAP
        shap_explanation = shap_explainer.get_sample_explanation(txn_idx_in_shap)
        shap_vals = shap_explanation['shap_values']
        
        # LIME
        idx_in_xtest = list(X_test.index).index(txn_idx_comp) if txn_idx_comp in X_test.index else 0
        lime_explanation = lime_explainer.explain_instance(X_test, idx_in_xtest, num_features=len(feature_names))
    
    # Graphiques côte à côte
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### SHAP Values")
        
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': shap_vals
        }).sort_values('Contribution', key=abs, ascending=True)
        
        fig_shap_comp = px.bar(
            shap_df,
            x='Contribution',
            y='Feature',
            orientation='h',
            color='Contribution',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title='Contributions SHAP'
        )
        fig_shap_comp.update_layout(height=450)
        st.plotly_chart(fig_shap_comp, use_container_width=True)
    
    with col2:
        st.markdown("### LIME Values")
        
        lime_df = lime_explainer.get_explanation_df(lime_explanation, label=1)
        lime_df = lime_df.sort_values('Weight', key=abs, ascending=True)
        
        fig_lime_comp = px.bar(
            lime_df,
            x='Weight',
            y='Feature',
            orientation='h',
            color='Weight',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title='Contributions LIME'
        )
        fig_lime_comp.update_layout(height=450)
        st.plotly_chart(fig_lime_comp, use_container_width=True)
    
    # Comparaison des rankings
    st.markdown("### Comparaison des rankings")
    
    from xai_utils import compare_explanations
    comparison_df, correlation = compare_explanations(shap_vals, lime_explanation, feature_names)
    
    # Afficher les métriques de concordance
    col1, col2, col3 = st.columns(3)
    
    # Top 5 features par méthode
    shap_ranking = comparison_df.nlargest(5, '|SHAP|')['Feature'].tolist()
    lime_ranking = comparison_df.nlargest(5, '|LIME|')['Feature'].tolist()
    common_top5 = set(shap_ranking).intersection(set(lime_ranking))
    
    with col1:
        st.metric("Corrélation", f"{correlation:.2f}" if not np.isnan(correlation) else "N/A")
    with col2:
        st.metric("Top 5 en commun", f"{len(common_top5)}/5")
    with col3:
        agreement = "Forte" if len(common_top5) >= 4 else "Modérée" if len(common_top5) >= 2 else "Faible"
        st.metric("Concordance", agreement)
    
    # Scatter plot de corrélation
    st.markdown("### Corrélation entre SHAP et LIME")
    
    fig_corr = px.scatter(
        comparison_df,
        x='|SHAP|',
        y='|LIME|',
        text='Feature',
        title=f'Corrélation SHAP vs LIME (r = {correlation:.3f})' if not np.isnan(correlation) else 'Corrélation SHAP vs LIME'
    )
    fig_corr.update_traces(textposition='top center')
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    correlation_str = f"{correlation:.2f}" if not np.isnan(correlation) else "N/A"
    st.info(f"""
    **Features dans le Top 5 des deux méthodes:** {', '.join(common_top5) if common_top5 else 'Aucune'}
    
    **Interprétation:**
    - Corrélation de {correlation_str} entre les valeurs absolues
    - {len(common_top5)} features sur 5 sont dans le top des deux méthodes
    - Les deux méthodes {'concordent' if len(common_top5) >= 3 else 'different'} sur les features les plus importantes
    """)

# =========================================
# Footer
# =========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    <p><b>Note:</b> Les explications sont calculées a partir des vrais modeles entrainés.</p>
</div>
""", unsafe_allow_html=True)
