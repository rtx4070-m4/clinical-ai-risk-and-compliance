
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report

def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:,1]
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall = tp/(tp+fn+1e-9)
    roc = roc_auc_score(y_test, proba)
    return {
        "cm": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": recall,
        "roc_auc": roc,
        "report": classification_report(y_test, y_pred)
    }

def tune_threshold_for_recall(model, X_val, y_val, target_recall=0.9):
    proba = model.predict_proba(X_val)[:,1]
    best_t, best_gap = 0.5, 1.0
    for t in np.linspace(0.05,0.95,19):
        y_pred = (proba>=t).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall = tp/(tp+fn+1e-9)
        gap = abs(target_recall - recall)
        if gap < best_gap:
            best_gap, best_t = gap, t
    return best_t
