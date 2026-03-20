
import numpy as np

def find_false_negatives(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:,1]
    pred = (proba>=threshold).astype(int)
    idx = [i for i in range(len(y)) if y[i]==1 and pred[i]==0]
    return idx

def harm_score(X, indices):
    # simple proxy: higher creatinine & BP => higher harm if missed
    cols = list(X.columns)
    c_idx = cols.index("creatinine")
    bp_idx = cols.index("bp")
    scores = []
    for i in indices:
        scores.append(0.7*abs(X.iloc[i, c_idx]) + 0.3*abs(X.iloc[i, bp_idx]))
    return float(np.mean(scores)) if scores else 0.0
