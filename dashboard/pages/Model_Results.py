"""
Model Results
================
Page d'affichage des résultats des modèles de détection de fraude.
Utilise les vrais modèles sauvegardés.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Configuration de la page
st.set_page_config(
    page_title="Model Results - Fraud Detection",
    page_icon="",
    layout="wide"
)

# Titre
st.markdown("# Model Results")
st.markdown("Comparaison des performances des différents modèles de détection de fraude.")

# =========================================
# Chargement des données et modèles
# =========================================
@st.cache_data
def load_data():
    """Charge les données de test depuis les fichiers parquet."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_path, "data")
    
    try:
        X_test = pd.read_parquet(os.path.join(data_path, "X_test.parquet"))
        y_test = pd.read_parquet(os.path.join(data_path, "y_test.parquet"))
        return X_test, y_test, True
    except FileNotFoundError as e:
        return None, None, False

@st.cache_resource
def load_models():
    """Charge tous les modèles sauvegardés."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_path = os.path.join(base_path, "models")
    
    model_files = {
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }
    
    models = {}
    for name, filename in model_files.items():
        filepath = os.path.join(models_path, filename)
        if os.path.exists(filepath):
            try:
                models[name] = joblib.load(filepath)
            except Exception as e:
                st.warning(f"Erreur lors du chargement de {name}: {e}")
    
    return models

@st.cache_data
def compute_model_metrics(_models, _X_test, _y_test):
    """Calcule les métriques pour chaque modèle."""
    results = {}
    
    y_test_array = _y_test.values.ravel() if hasattr(_y_test, 'values') else _y_test
    
    for name, model in _models.items():
        try:
            y_pred = model.predict(_X_test)
            y_proba = model.predict_proba(_X_test)[:, 1]
            
            cm = confusion_matrix(y_test_array, y_pred)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(_X_test.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(_X_test.columns, np.abs(model.coef_[0])))
            else:
                importance = {col: 1/len(_X_test.columns) for col in _X_test.columns}
            
            results[name] = {
                'accuracy': accuracy_score(y_test_array, y_pred),
                'precision': precision_score(y_test_array, y_pred, zero_division=0),
                'recall': recall_score(y_test_array, y_pred, zero_division=0),
                'f1_score': f1_score(y_test_array, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test_array, y_proba),
                'confusion_matrix': cm,
                'feature_importance': importance,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        except Exception as e:
            st.warning(f"Erreur lors de l'évaluation de {name}: {e}")
    
    return results

@st.cache_data
def compute_roc_curves(_models, _X_test, _y_test):
    """Calcule les courbes ROC pour chaque modèle."""
    roc_data = {}
    
    y_test_array = _y_test.values.ravel() if hasattr(_y_test, 'values') else _y_test
    
    for name, model in _models.items():
        try:
            y_proba = model.predict_proba(_X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_array, y_proba)
            auc = roc_auc_score(y_test_array, y_proba)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        except Exception as e:
            st.warning(f"Erreur calcul ROC pour {name}: {e}")
    
    return roc_data

# Charger les données
X_test, y_test, data_loaded = load_data()

if not data_loaded:
    st.error("Les données de test n'ont pas pu être chargées.")
    st.info("Fichiers attendus: data/X_test.parquet, data/y_test.parquet")
    st.stop()

# Charger les modèles
models = load_models()

if not models:
    st.error("Aucun modèle n'a pu être chargé.")
    st.info("""
    Assurez-vous d'avoir sauvegardé les modèles dans le dossier models/ :
    - models/random_forest.pkl
    - models/xgboost.pkl  
    - models/lightgbm.pkl
    """)
    st.code("""
# Dans votre notebook, sauvegardez les modèles ainsi:
import joblib
joblib.dump(rf_model, '../models/random_forest.pkl')
joblib.dump(xgb_model, '../models/xgboost.pkl')
joblib.dump(lgb_model, '../models/lightgbm.pkl')
    """)
    st.stop()

# Calculer les métriques
with st.spinner("Calcul des métriques..."):
    results = compute_model_metrics(models, X_test, y_test)
    roc_data = compute_roc_curves(models, X_test, y_test)

st.success(f"{len(results)} modèle(s) chargé(s) et évalué(s) avec succès!")

# Sidebar pour la sélection du modèle
st.sidebar.markdown("## Sélection")
selected_model = st.sidebar.selectbox(
    "Choisir un modèle",
    list(results.keys())
)

view_mode = st.sidebar.radio(
    "Mode d'affichage",
    ["Comparaison globale", "Détail d'un modèle"]
)

# =========================================
# Vue 1: Comparaison globale
# =========================================
if view_mode == "Comparaison globale":
    st.markdown("---")
    st.markdown("## Comparaison des Modèles")
    
    # Créer un DataFrame de comparaison
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Modèle': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Afficher le tableau
    st.markdown("### Tableau récapitulatif")
    
    # Formater le DataFrame pour l'affichage
    styled_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Identifier le meilleur modèle
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Modèle']
    best_f1 = comparison_df['F1-Score'].max()
    
    st.success(f"**Meilleur modèle (F1-Score):** {best_model} avec {best_f1:.2%}")
    
    # Graphiques de comparaison
    st.markdown("### Visualisation des métriques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart des métriques principales
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        fig_metrics = go.Figure()
        
        colors = px.colors.qualitative.Set2
        for i, model in enumerate(comparison_df['Modèle']):
            values = [comparison_df.loc[comparison_df['Modèle'] == model, m].values[0] for m in metrics_to_plot]
            fig_metrics.add_trace(go.Bar(
                name=model,
                x=metrics_to_plot,
                y=values,
                marker_color=colors[i % len(colors)]
            ))
        
        fig_metrics.update_layout(
            title='Comparaison des métriques par modèle',
            barmode='group',
            yaxis_title='Score',
            yaxis_range=[0, 1.05],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # Radar chart
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig_radar = go.Figure()
        
        for i, model in enumerate(comparison_df['Modèle']):
            values = [comparison_df.loc[comparison_df['Modèle'] == model, m].values[0] for m in categories]
            values.append(values[0])  # Fermer le polygone
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model,
                opacity=0.6
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
            showlegend=True,
            title='Radar des performances',
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Courbes ROC
    st.markdown("### Courbes ROC")
    
    fig_roc = go.Figure()
    
    # Ligne de référence (random classifier)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='gray', dash='dash')
    ))
    
    colors = px.colors.qualitative.Set1
    for i, (model, data) in enumerate(roc_data.items()):
        fig_roc.add_trace(go.Scatter(
            x=data['fpr'],
            y=data['tpr'],
            mode='lines',
            name=f"{model} (AUC = {data['auc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig_roc.update_layout(
        title='Courbes ROC - Comparaison des modèles',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        legend=dict(x=0.6, y=0.1),
        height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# =========================================
# Vue 2: Détail d'un modèle
# =========================================
else:
    st.markdown("---")
    st.markdown(f"## Détails: {selected_model}")
    
    model_data = results[selected_model]
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{model_data['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{model_data['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{model_data['recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{model_data['f1_score']:.2%}")
    with col5:
        st.metric("ROC-AUC", f"{model_data['roc_auc']:.3f}")
    
    # Matrice de confusion
    st.markdown("### Matrice de Confusion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cm = model_data['confusion_matrix']
        
        # Heatmap de la matrice de confusion
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Prédit", y="Réel", color="Nombre"),
            x=['Non-Fraude', 'Fraude'],
            y=['Non-Fraude', 'Fraude'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(title=f'Matrice de Confusion - {selected_model}')
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Métriques détaillées de la matrice
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown("#### Détails de la matrice")
        
        metrics_detail = pd.DataFrame({
            'Métrique': ['True Negatives (TN)', 'False Positives (FP)', 
                        'False Negatives (FN)', 'True Positives (TP)',
                        'Total Prédictions'],
            'Valeur': [f"{tn:,}", f"{fp:,}", f"{fn:,}", f"{tp:,}", f"{tn+fp+fn+tp:,}"],
            'Description': [
                'Transactions légitimes correctement classées',
                'Fausses alertes (légitimes classées fraude)',
                'Fraudes non détectées',
                'Fraudes correctement détectées',
                'Nombre total de prédictions'
            ]
        })
        st.dataframe(metrics_detail, use_container_width=True, hide_index=True)
        
        # Calculs supplémentaires
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        st.info(f"""
        **Métriques supplémentaires:**
        - Spécificité (TNR): {specificity:.2%}
        - Valeur Prédictive Négative: {npv:.2%}
        - Taux de fausse alerte: {fp/(tn+fp):.2%}
        """)
    
    # Feature Importance
    st.markdown("### Feature Importance")
    
    importance_df = pd.DataFrame({
        'Feature': list(model_data['feature_importance'].keys()),
        'Importance': list(model_data['feature_importance'].values())
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Feature Importance - {selected_model}',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Courbe ROC individuelle
    st.markdown("### Courbe ROC")
    
    if selected_model in roc_data:
        roc_model = roc_data[selected_model]
        
        fig_roc_single = go.Figure()
        
        fig_roc_single.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.5)',
            line=dict(color='gray', dash='dash')
        ))
        
        fig_roc_single.add_trace(go.Scatter(
            x=roc_model['fpr'],
            y=roc_model['tpr'],
            mode='lines',
            name=f"{selected_model} (AUC = {roc_model['auc']:.3f})",
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=3)
        ))
        
        fig_roc_single.update_layout(
            title=f'Courbe ROC - {selected_model}',
            xaxis_title='False Positive Rate (FPR)',
            yaxis_title='True Positive Rate (TPR)',
            height=400
        )
        st.plotly_chart(fig_roc_single, use_container_width=True)
    
    # Classification Report
    st.markdown("### Classification Report")
    
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    report_df = pd.DataFrame({
        'Classe': ['Non-Fraude (0)', 'Fraude (1)', 'Macro Avg', 'Weighted Avg'],
        'Precision': [f"{precision_0:.2%}", f"{precision_1:.2%}", 
                     f"{(precision_0+precision_1)/2:.2%}", f"{model_data['precision']:.2%}"],
        'Recall': [f"{recall_0:.2%}", f"{recall_1:.2%}",
                  f"{(recall_0+recall_1)/2:.2%}", f"{model_data['recall']:.2%}"],
        'F1-Score': [f"{f1_0:.2%}", f"{f1_1:.2%}",
                    f"{(f1_0+f1_1)/2:.2%}", f"{model_data['f1_score']:.2%}"],
        'Support': [f"{tn+fp:,}", f"{fn+tp:,}", f"{tn+fp+fn+tp:,}", f"{tn+fp+fn+tp:,}"]
    })
    
    st.dataframe(report_df, use_container_width=True, hide_index=True)

# =========================================
# Téléchargement des résultats
# =========================================
st.markdown("---")
st.markdown("## Exporter les résultats")

col1, col2 = st.columns(2)

with col1:
    # Export comparaison en CSV
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Modèle': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    export_df = pd.DataFrame(comparison_data)
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="Télécharger comparaison (CSV)",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

with col2:
    # Export feature importance
    all_importance = []
    for model_name, metrics in results.items():
        for feature, importance in metrics['feature_importance'].items():
            all_importance.append({
                'Modèle': model_name,
                'Feature': feature,
                'Importance': importance
            })
    
    importance_export_df = pd.DataFrame(all_importance)
    csv_importance = importance_export_df.to_csv(index=False)
    
    st.download_button(
        label="Télécharger feature importance (CSV)",
        data=csv_importance,
        file_name="feature_importance.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    <p>Résultats calculés à partir des vrais modèles entraînés sur le test set.</p>
</div>
""", unsafe_allow_html=True)