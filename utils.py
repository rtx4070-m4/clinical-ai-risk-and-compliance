
import joblib
import numpy as np
import pandas as pd

def load_bundle(path="models/best_model.joblib"):
    return joblib.load(path)

def transform(bundle, row_dict):
    cols = bundle["columns"]
    X = pd.DataFrame([row_dict])[cols]
    X['rbc'] = X['rbc'].map({'normal':0,'abnormal':1})
    X['hypertension'] = X['hypertension'].map({'no':0,'yes':1})
    X = pd.DataFrame(bundle["imputer"].transform(X), columns=cols)
    X = pd.DataFrame(bundle["scaler"].transform(X), columns=cols)
    X = bundle["pca"].transform(X)
    return X

def predict(bundle, X):
    proba = bundle["model"].predict_proba(X)[:,1]
    pred = (proba >= bundle["threshold"]).astype(int)
    return float(proba[0]), int(pred[0])
