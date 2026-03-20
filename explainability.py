
import shap
import numpy as np
import lime.lime_tabular

def shap_global(model, X):
    explainer = shap.Explainer(model, X)
    values = explainer(X)
    return values

def shap_local(model, X, idx):
    explainer = shap.Explainer(model, X)
    values = explainer(X)
    return values[idx]

def lime_local(model, X_train, X_instance):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=list(X_train.columns),
        class_names=["No CKD","CKD"], discretize_continuous=True
    )
    exp = explainer.explain_instance(
        X_instance.values, model.predict_proba, num_features=6
    )
    return exp
