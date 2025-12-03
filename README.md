## A. Feature Engineering & Sélection de Variables

### 1. Sélection de variables
- RFE (Recursive Feature Elimination).
- Mutual Information.
- Sélection via SHAP (pruning).
- Analyse des corrélations & élimination des redondances.

### 2. Réduction de dimension
- PCA pour représentations compactes.
- t-SNE / UMAP pour visualisation des anomalies.
- Comparaison des performances avant/après réduction.

---

## B. Modélisation (Supervisée & Non Supervisée)

### 3. Gestion du déséquilibre des classes
- SMOTE / ADASYN.
- Undersampling intelligent.
- Optimisation du threshold de décision.
- Analyse Precision-Recall pour prioriser les cas suspects.

### 4. Nouveaux modèles à tester
- **Supervisés** : Random Forest, LightGBM, MLP.
- **Non-supervisés** : LOF, OCSVM, Isolation Forest.
- Comparaison systématique des métriques (Recall, F1).

### 5. Hyperparameter Tuning
- GridSearch, RandomizedSearch ou Optuna.
- Optimisation spécialisée pour données bruitées et déséquilibrées.

### 6. Méta-modélisation (Stacking / Ensemble)
- Stacking avec combinaisons XGBoost + RF + OCSVM + LOF.
- Tests de blending et soft voting.
- Évaluer robustesse et stabilité du méta-modèle.

---

## C. Robustesse & Validation

### 7. Validation croisée avancée
- Stratified k-fold.
- Tests sur jeux perturbés (ajout de bruit contrôlé).
- Validation temporelle (time-based split si applicable).

---

## D. Approches Semi-Supervisées

### 8. Semi-supervisé & weak labels
- Label Propagation.
- Pseudo-labelling sur modèles supervisés performants.
- Analyse de confiance des labels générés automatiquement.

---

## E. Explicabilité (XAI)

### 9. SHAP
- SHAP global (summary plot).
- SHAP local (explication de chaque cas suspect).
- SHAP pour modèles du stacking (TreeExplainer).

### 10. LIME & comparaison d’explicabilité
- Interprétation locale sur transactions borderline.
- Comparaison lisibilité / robustesse SHAP vs LIME.
- Construction d’un rapport « Explainability for Auditors ».

---

## F. Industrialisation & Prototype

### 11. Pipeline Python reproductible
- Architecture en modules :  
  `preprocessing/`, `models/`, `evaluation/`, `xai/`.
- Fichier de configuration (`config.yaml`).
- Logging + suivi des paramètres.

### 12. Dashboard Streamlit/Dash
- Import des données par l’auditeur.
- Visualisation des anomalies.
- Priorisation des cas suspects.
- Explicabilité intégrée (SHAP plots interactifs).

---

## F. Documentation & Valorisation

### 13. Documentation GitHub
- README structuré.
- Guide d’installation.
- Exemple d’exécution du pipeline.
- Organisation des notebooks (EDA / Modèles / XAI / Dashboard).

### 14. Rédaction de l’article scientifique
- Introduction & Related Work.
- Méthodologie du méta-modèle hybride.
- Expériences & résultats.
- Illustrations (figures, SHAP, architecture du pipeline).

---