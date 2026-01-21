"""
dashboard/xai_utils.py
======================
Utilitaires XAI (SHAP + LIME) pour modèles tabulaires.
AUCUN code Streamlit ici.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

RANDOM_STATE = 42


def safe_predict_proba_pos(model, X: pd.DataFrame) -> np.ndarray:
    """Retourne p(classe=1) pour chaque ligne."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
        raise ValueError("predict_proba a renvoyé une forme inattendue.")

    pred = np.asarray(model.predict(X))
    if pred.ndim == 2 and pred.shape[1] >= 2:
        return pred[:, 1]
    if pred.ndim == 1:
        return pred
    raise ValueError("Impossible d'obtenir une probabilité depuis ce modèle.")


def to_numpy_2d_shap(shap_values) -> np.ndarray:
    """Normalise SHAP vers (n_samples, n_features)."""
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # certains explainers renvoient une liste [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        shap_values = shap_values[1]  # classe positive

    arr = np.array(shap_values)

    # parfois (n, f, classes)
    if arr.ndim == 3:
        arr = arr[:, :, 1]

    if arr.ndim != 2:
        raise ValueError(f"SHAP values inattendues: ndim={arr.ndim}")

    return arr


class SHAPExplainer:
    """Wrapper SHAP TreeExplainer robuste (RF/XGB/LGBM/CatBoost) avec optimisations de performance."""

    def __init__(self, model, X_background: pd.DataFrame, feature_names: list[str], max_trees: int | None = None):
        import shap  # lazy import

        self.model = model
        self.feature_names = feature_names
        
        # Détection Random Forest pour optimisations spécifiques
        model_type = str(type(model).__name__).lower()
        self.is_random_forest = 'random' in model_type or 'forest' in model_type
        
        # OPTIMISATION CRITIQUE pour Random Forest : Limiter le nombre d'arbres
        # RF avec 100+ arbres peut prendre plusieurs minutes par échantillon
        if self.is_random_forest and max_trees is not None:
            try:
                # Cloner le modèle en ne gardant que les premiers arbres
                import copy
                model_copy = copy.deepcopy(model)
                if hasattr(model_copy, 'estimators_'):
                    original_count = len(model_copy.estimators_)
                    if original_count > max_trees:
                        model_copy.estimators_ = model_copy.estimators_[:max_trees]
                        model_copy.n_estimators = max_trees
                        self.model = model_copy
                        print(f"⚡ RF optimisé : {original_count} → {max_trees} arbres")
            except Exception as e:
                print(f"⚠️ Impossible de limiter les arbres RF: {e}")
        
        # OPTIMISATION : Pour Random Forest, utiliser interventional avec un background dataset plus large
        # tree_path_dependent nécessite que le background couvre toutes les feuilles du modèle
        if self.is_random_forest:
            # Utiliser un background plus large pour RF (100 samples) avec interventional
            bg_size = min(100, len(X_background))
            self.explainer = shap.TreeExplainer(
                self.model,
                data=X_background[feature_names].head(bg_size),
                feature_perturbation='interventional',  # Plus robuste que tree_path_dependent
                model_output='raw'  # évite les conversions inutiles
            )
        else:
            self.explainer = shap.TreeExplainer(self.model)

        self.X_explained: pd.DataFrame | None = None
        self.shap_values: np.ndarray | None = None

    def compute(self, X: pd.DataFrame, max_samples: int | None = None) -> np.ndarray:
        X_use = X[self.feature_names].copy()
        if max_samples is not None and len(X_use) > max_samples:
            X_use = X_use.sample(n=max_samples, random_state=RANDOM_STATE)

        self.X_explained = X_use.reset_index(drop=True)
        
        # OPTIMISATION CRITIQUE : Désactiver check_additivity pour gagner 30-50% de temps
        vals = self.explainer.shap_values(
            self.X_explained, 
            check_additivity=False
        )
        self.shap_values = to_numpy_2d_shap(vals)
        return self.shap_values

    def feature_importance(self) -> pd.DataFrame:
        if self.shap_values is None:
            raise ValueError("SHAP non calculé.")
        imp = np.abs(self.shap_values).mean(axis=0).ravel()
        return (
            pd.DataFrame({"Feature": self.feature_names, "Importance": imp})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    def local(self, sample_idx: int) -> dict:
        if self.shap_values is None or self.X_explained is None:
            raise ValueError("SHAP non calculé.")

        base = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(base, np.ndarray):
            base = float(base[-1])

        return {
            "base_value": float(base),
            "shap_values": np.array(self.shap_values[sample_idx]).ravel(),
            "x": self.X_explained.iloc[sample_idx],
        }


class LIMEExplainer:
    """Wrapper LIME tabulaire robuste."""

    def __init__(self, model, X_train: pd.DataFrame, feature_names: list[str]):
        from lime import lime_tabular  # lazy import

        self.model = model
        self.feature_names = feature_names
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train[feature_names].values,
            feature_names=feature_names,
            class_names=["Non-Fraud", "Fraud"],
            mode="classification",
            random_state=RANDOM_STATE,
        )

    def explain(self, X: pd.DataFrame, sample_pos_idx: int, num_features: int = 10):
        row = X[self.feature_names].iloc[sample_pos_idx].values
        return self.explainer.explain_instance(
            data_row=row,
            predict_fn=lambda arr: self.model.predict_proba(pd.DataFrame(arr, columns=self.feature_names)),
            num_features=num_features,
        )

    @staticmethod
    def to_df(explanation, label: int = 1) -> pd.DataFrame:
        return pd.DataFrame(explanation.as_list(label=label), columns=["Feature", "Weight"])


def compare_shap_lime(shap_1d: np.ndarray, lime_exp, feature_names: list[str]) -> tuple[pd.DataFrame, float]:
    shap_1d = np.asarray(shap_1d).ravel()

    lime_map = {}
    for rule, w in lime_exp.as_list(label=1):
        for fn in feature_names:
            if fn in rule:
                lime_map[fn] = float(w)
                break

    df = pd.DataFrame({"Feature": feature_names, "SHAP": shap_1d})
    df["LIME"] = df["Feature"].map(lime_map).fillna(0.0)
    df["|SHAP|"] = np.abs(df["SHAP"])
    df["|LIME|"] = np.abs(df["LIME"])

    corr = np.corrcoef(df["|SHAP|"], df["|LIME|"])[0, 1] if len(df) > 1 else np.nan
    return df, (float(corr) if not np.isnan(corr) else np.nan)
