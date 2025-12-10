"""
XAI Utilities for Fraud Detection
=================================
Classes réutilisables SHAP et LIME extraites du notebook XAI.ipynb
"""

import numpy as np
import pandas as pd
import shap
from lime import lime_tabular

RANDOM_STATE = 42


class SHAPExplainer:
    """
    Reusable SHAP explainer for fraud detection models.
    """
    
    def __init__(self, model, X_background, feature_names=None, model_type='tree'):
        """
        Initialize SHAP explainer.
        
        Parameters:
        -----------
        model : trained model
            The model to explain
        X_background : pd.DataFrame or np.array
            Background data for SHAP (sample of training data)
        feature_names : list
            Names of features
        model_type : str
            Type of explainer ('tree' for tree-based models, 'kernel' for others)
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names if feature_names is not None else list(X_background.columns)
        self.model_type = model_type
        
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'kernel':
            background_sample = shap.sample(X_background, min(100, len(X_background)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.shap_values = None
        self.X_explained = None
    
    def compute_shap_values(self, X, max_samples=None):
        """
        Compute SHAP values for a dataset.
        """
        if max_samples is not None and len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=RANDOM_STATE)
        
        self.X_explained = X.reset_index(drop=True)
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, extract positive class SHAP values
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        # Handle SHAP Explanation object
        if hasattr(self.shap_values, 'values'):
            self.shap_values = self.shap_values.values
        
        # Ensure numpy array
        self.shap_values = np.array(self.shap_values)
        
        # Handle 3D array (samples x features x classes) for binary classification
        if self.shap_values.ndim == 3:
            self.shap_values = self.shap_values[:, :, 1]  # Take positive class
        
        return self.shap_values
    
    def get_feature_importance(self):
        """
        Get mean absolute SHAP values for feature importance.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet.")
        
        shap_vals = self.shap_values
        
        # Handle SHAP Explanation object
        if hasattr(shap_vals, 'values'):
            shap_vals = shap_vals.values
        
        # Ensure numpy array
        shap_vals = np.array(shap_vals)
        
        # Handle 3D array (samples x features x classes) for binary classification
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]  # Take positive class
        
        # Calculate mean absolute SHAP values per feature
        mean_shap = np.abs(shap_vals).mean(axis=0)
        
        # Ensure 1D
        mean_shap = np.array(mean_shap).ravel()
        
        # Match lengths - truncate or pad if needed
        n_features = len(self.feature_names)
        if len(mean_shap) > n_features:
            mean_shap = mean_shap[:n_features]
        elif len(mean_shap) < n_features:
            mean_shap = np.pad(mean_shap, (0, n_features - len(mean_shap)), constant_values=0)
        
        importance_df = pd.DataFrame({
            'Feature': list(self.feature_names),
            'Importance': list(mean_shap)
        }).sort_values('Importance', ascending=False)
        
        return importance_df
        
        return importance_df
    
    def get_sample_explanation(self, sample_idx):
        """
        Get SHAP explanation for a single sample.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet.")
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        # Ensure shap_values for single sample is 1D
        sample_shap = np.array(self.shap_values[sample_idx]).ravel()
        
        return {
            'shap_values': sample_shap,
            'base_value': base_value,
            'feature_values': self.X_explained.iloc[sample_idx].values,
            'feature_names': self.feature_names
        }
    
    def get_shap_explanation_object(self, sample_idx):
        """
        Get a SHAP Explanation object for waterfall/force plots.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet.")
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        # Ensure shap_values for single sample is 1D
        sample_shap = np.array(self.shap_values[sample_idx]).ravel()
        
        return shap.Explanation(
            values=sample_shap,
            base_values=base_value,
            data=self.X_explained.iloc[sample_idx].values,
            feature_names=self.feature_names
        )


class LIMEExplainer:
    """
    Reusable LIME explainer for fraud detection models.
    """
    
    def __init__(self, model, X_train, feature_names=None, class_names=None):
        """
        Initialize LIME explainer.
        
        Parameters:
        -----------
        model : trained model
            The model to explain
        X_train : pd.DataFrame or np.array
            Training data (used to understand feature distributions)
        feature_names : list
            Names of features
        class_names : list
            Names of classes
        """
        self.model = model
        self.feature_names = feature_names if feature_names is not None else list(X_train.columns)
        self.class_names = class_names if class_names is not None else ['Non-Fraud', 'Fraud']
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=RANDOM_STATE
        )
    
    def explain_instance(self, X, sample_idx, num_features=10):
        """
        Explain a single prediction using LIME.
        
        Returns:
        --------
        explanation : lime.explanation.Explanation
            LIME explanation object
        """
        if isinstance(X, pd.DataFrame):
            sample = X.iloc[sample_idx].values
        else:
            sample = X[sample_idx]
        
        explanation = self.explainer.explain_instance(
            data_row=sample,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )
        
        return explanation
    
    def get_explanation_df(self, explanation, label=1):
        """
        Convert LIME explanation to a DataFrame.
        
        Parameters:
        -----------
        explanation : lime.explanation.Explanation
            LIME explanation object
        label : int
            Class label to explain (1 for fraud)
        
        Returns:
        --------
        pd.DataFrame with features and weights
        """
        exp_list = explanation.as_list(label=label)
        return pd.DataFrame(exp_list, columns=['Feature', 'Weight'])


def compare_explanations(shap_values, lime_explanation, feature_names):
    """
    Compare SHAP and LIME explanations.
    
    Returns DataFrame with both explanations and correlation.
    """
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Feature': feature_names,
        'SHAP': shap_values
    })
    
    # Add LIME weights (need to extract feature name from LIME format)
    lime_dict = {}
    lime_list = lime_explanation.as_list(label=1)
    for feat, weight in lime_list:
        for fn in feature_names:
            if fn in feat:
                lime_dict[fn] = weight
                break
    
    comparison['LIME'] = comparison['Feature'].map(lime_dict).fillna(0)
    comparison['|SHAP|'] = np.abs(comparison['SHAP'])
    comparison['|LIME|'] = np.abs(comparison['LIME'])
    
    # Calculate correlation
    correlation = np.corrcoef(comparison['|SHAP|'], comparison['|LIME|'])[0, 1]
    
    return comparison, correlation
