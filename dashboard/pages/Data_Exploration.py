"""
Data Exploration
================
Page de visualisation et d'exploration du dataset de fraude financière.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration de la page
st.set_page_config(
    page_title="Data Exploration - Fraud Detection",
    page_icon="",
    layout="wide"
)

# Titre
st.markdown("# Data Exploration")
st.markdown("Exploration et visualisation du dataset de transactions financières.")

# Fonction de chargement des données avec cache
@st.cache_data
def load_data():
    """Charge le dataset depuis le fichier parquet."""
    # Chemin relatif depuis le dossier pages -> dashboard -> racine projet -> data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "df_allcolumns.parquet")
    
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        return df
    else:
        st.error(f"Fichier non trouvé : {data_path}")
        return None

# Chargement des données
with st.spinner("Chargement des données..."):
    df = load_data()

if df is not None:
    # Sidebar pour les filtres
    st.sidebar.markdown("## Filtres")
    
    # Filtre par type de transaction (si la colonne existe)
    if 'type' in df.columns:
        transaction_types = ['Tous'] + list(df['type'].unique())
        selected_type = st.sidebar.selectbox("Type de transaction", transaction_types)
        
        if selected_type != 'Tous':
            df_filtered = df[df['type'] == selected_type]
        else:
            df_filtered = df
    else:
        df_filtered = df
    
    # Filtre par montant
    if 'amount' in df.columns:
        min_amount = float(df['amount'].min())
        max_amount = float(df['amount'].max())
        amount_range = st.sidebar.slider(
            "Plage de montant",
            min_value=min_amount,
            max_value=max_amount,
            value=(min_amount, max_amount)
        )
        df_filtered = df_filtered[(df_filtered['amount'] >= amount_range[0]) & 
                                   (df_filtered['amount'] <= amount_range[1])]
    
    # Nombre d'échantillons à afficher
    sample_size = st.sidebar.slider("Échantillon à afficher", 100, 10000, 1000)
    
    # =========================================
    # Section 1: Overview du dataset
    # =========================================
    st.markdown("---")
    st.markdown("## Overview du Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de lignes", f"{len(df):,}")
    
    with col2:
        st.metric("Nombre de colonnes", f"{len(df.columns)}")
    
    with col3:
        if 'isFraud' in df.columns:
            fraud_rate = (df['isFraud'].sum() / len(df)) * 100
            st.metric("Taux de fraude", f"{fraud_rate:.2f}%")
        else:
            st.metric("Taux de fraude", "N/A")
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Mémoire utilisée", f"{memory_mb:.1f} MB")
    
    # Affichage de l'échantillon des données
    st.markdown("### Aperçu des données")
    st.dataframe(df_filtered.head(sample_size), use_container_width=True)
    
    # Informations sur les colonnes
    with st.expander("Informations sur les colonnes"):
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Statistiques descriptives
    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe(), use_container_width=True)
    
    # =========================================
    # Section 2: Distribution de la cible
    # =========================================
    st.markdown("---")
    st.markdown("## Distribution de la Variable Cible (Fraude)")
    
    if 'isFraud' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fraud_counts = df['isFraud'].value_counts()
            fig_pie = px.pie(
                values=fraud_counts.values,
                names=['Non-Fraude (0)', 'Fraude (1)'],
                title='Répartition Fraude / Non-Fraude',
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=['Non-Fraude', 'Fraude'],
                y=fraud_counts.values,
                title='Nombre de transactions par classe',
                color=['Non-Fraude', 'Fraude'],
                color_discrete_map={'Non-Fraude': '#2ecc71', 'Fraude': '#e74c3c'},
                text=fraud_counts.values
            )
            fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Statistiques de déséquilibre
        st.info(f"""
        **Analyse du déséquilibre des classes:**
        - Transactions légitimes: **{fraud_counts.get(0, 0):,}** ({100 - fraud_rate:.2f}%)
        - Transactions frauduleuses: **{fraud_counts.get(1, 0):,}** ({fraud_rate:.2f}%)
        - Ratio d'imbalance: **1:{int(fraud_counts.get(0, 1) / max(fraud_counts.get(1, 1), 1))}**
        """)
    
    # =========================================
    # Section 3: Distribution des variables
    # =========================================
    st.markdown("---")
    st.markdown("## Distribution des Variables")
    
    # Sélection de la variable à visualiser
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_var = st.selectbox("Sélectionner une variable", numeric_cols)
        show_by_fraud = st.checkbox("Séparer par fraude", value=True)
        chart_type = st.radio("Type de graphique", ["Histogramme", "Boxplot"])
    
    with col2:
        if selected_var:
            if chart_type == "Histogramme":
                if show_by_fraud and 'isFraud' in df.columns:
                    fig = px.histogram(
                        df_filtered.sample(min(sample_size, len(df_filtered))),
                        x=selected_var,
                        color='isFraud',
                        barmode='overlay',
                        title=f'Distribution de {selected_var}',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                        opacity=0.7
                    )
                else:
                    fig = px.histogram(
                        df_filtered.sample(min(sample_size, len(df_filtered))),
                        x=selected_var,
                        title=f'Distribution de {selected_var}',
                        color_discrete_sequence=['#667eea']
                    )
            
            else:  # Boxplot
                if show_by_fraud and 'isFraud' in df.columns:
                    fig = px.box(
                        df_filtered.sample(min(sample_size, len(df_filtered))),
                        x='isFraud',
                        y=selected_var,
                        title=f'Boxplot de {selected_var}',
                        color='isFraud',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                    )
                else:
                    fig = px.box(
                        df_filtered.sample(min(sample_size, len(df_filtered))),
                        y=selected_var,
                        title=f'Boxplot de {selected_var}',
                        color_discrete_sequence=['#667eea']
                    )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================
    # Section 4: Analyse par type de transaction
    # =========================================
    if 'type' in df.columns:
        st.markdown("---")
        st.markdown("## Analyse par Type de Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des types
            type_counts = df['type'].value_counts()
            fig_types = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title='Nombre de transactions par type',
                color=type_counts.index,
                text=type_counts.values
            )
            fig_types.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_types.update_layout(showlegend=False, xaxis_title="Type", yaxis_title="Nombre")
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            if 'isFraud' in df.columns:
                # Taux de fraude par type
                fraud_by_type = df.groupby('type')['isFraud'].agg(['sum', 'count'])
                fraud_by_type['fraud_rate'] = (fraud_by_type['sum'] / fraud_by_type['count']) * 100
                
                fig_fraud_type = px.bar(
                    x=fraud_by_type.index,
                    y=fraud_by_type['fraud_rate'],
                    title='Taux de fraude par type de transaction (%)',
                    color=fraud_by_type['fraud_rate'],
                    color_continuous_scale='Reds',
                    text=fraud_by_type['fraud_rate'].round(2)
                )
                fig_fraud_type.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_fraud_type.update_layout(xaxis_title="Type", yaxis_title="Taux de fraude (%)")
                st.plotly_chart(fig_fraud_type, use_container_width=True)
    
    # =========================================
    # Section 5: Matrice de corrélation
    # =========================================
    st.markdown("---")
    st.markdown("## Matrice de Corrélation")
    
    # Sélection du nombre de features
    max_features = st.slider("Nombre de features à afficher", 5, min(30, len(numeric_cols)), 15)
    
    # Calculer la corrélation avec la cible si elle existe
    if 'isFraud' in df.columns:
        # Sélectionner les features les plus corrélées avec isFraud
        correlations = df[numeric_cols].corr(method='pearson')['isFraud'].abs().sort_values(ascending=False)
        top_features = correlations.head(max_features).index.tolist()
    else:
        top_features = numeric_cols[:max_features]
    
    # Calculer la matrice de corrélation
    corr_matrix = df[top_features].corr(method='pearson')
    
    # Heatmap avec Plotly
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Corrélation"),
        x=top_features,
        y=top_features,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Matrice de corrélation (Pearson)"
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Top corrélations avec la fraude
    if 'isFraud' in df.columns:
        st.markdown("### Top corrélations avec isFraud")
        top_corr_df = pd.DataFrame({
            'Feature': correlations.head(15).index,
            'Corrélation': correlations.head(15).values
        })
        
        fig_top_corr = px.bar(
            top_corr_df,
            x='Corrélation',
            y='Feature',
            orientation='h',
            title='Top 15 features corrélées avec isFraud',
            color='Corrélation',
            color_continuous_scale='Viridis'
        )
        fig_top_corr.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig_top_corr, use_container_width=True)
    
    # =========================================
    # Section 6: Scatter plots
    # =========================================
    st.markdown("---")
    st.markdown("## Scatter Plots")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_var = st.selectbox("Variable X", numeric_cols, index=0)
    with col2:
        y_var = st.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1))
    with col3:
        color_by_fraud = st.checkbox("Colorer par fraude", value=True, key="scatter_fraud")
    
    # Échantillonner pour performance
    sample_df = df_filtered.sample(min(5000, len(df_filtered)))
    
    if color_by_fraud and 'isFraud' in df.columns:
        fig_scatter = px.scatter(
            sample_df,
            x=x_var,
            y=y_var,
            color='isFraud',
            title=f'{x_var} vs {y_var}',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            opacity=0.6
        )
    else:
        fig_scatter = px.scatter(
            sample_df,
            x=x_var,
            y=y_var,
            title=f'{x_var} vs {y_var}',
            color_discrete_sequence=['#667eea'],
            opacity=0.6
        )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # =========================================
    # Section 7: Téléchargement
    # =========================================
    st.markdown("---")
    st.markdown("## Téléchargement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Télécharger un échantillon en CSV
        csv = df_filtered.head(10000).to_csv(index=False)
        st.download_button(
            label="Télécharger un échantillon (CSV)",
            data=csv,
            file_name="fraud_data_sample.csv",
            mime="text/csv"
        )
    
    with col2:
        # Télécharger les statistiques
        stats_csv = df.describe().to_csv()
        st.download_button(
            label="Télécharger les statistiques (CSV)",
            data=stats_csv,
            file_name="fraud_data_statistics.csv",
            mime="text/csv"
        )

else:
    st.error("Impossible de charger les données. Vérifiez que le fichier `df_allcolumns.parquet` existe dans le dossier `data/`.")
    st.info("Chemin attendu: `../data/df_allcolumns.parquet`")
