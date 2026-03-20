
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess
from src.feature_engineering import pca_reduce
from src.models import get_models
from src.evaluation import evaluate, tune_threshold_for_recall
from src.visualization import plot_cm, plot_roc

DATA_PATH = "data/raw/ckd_synthetic.csv"
MODEL_DIR = "models"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X, y, imp, scaler = preprocess(df)
    X_pca, pca = pca_reduce(X, n_components=5)

    X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    models = get_models()
    best = None
    best_recall = -1

    for name, m in models.items():
        m.fit(X_tr, y_tr)
        t = tune_threshold_for_recall(m, X_tr, y_tr, target_recall=0.9)
        metrics = evaluate(m, X_te, y_te, threshold=t)

        # save plots
        plot_cm(m, X_te, y_te, os.path.join(MODEL_DIR, f"{name}_cm.png"))
        plot_roc(m, X_te, y_te, os.path.join(MODEL_DIR, f"{name}_roc.png"))

        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            best = (name, m, t, metrics)

        print(f"Model: {name} | Recall: {metrics['recall']:.3f} | FN: {metrics['fn']} | ROC-AUC: {metrics['roc_auc']:.3f}")

    name, model, thr, metrics = best
    print(f"\nBEST MODEL: {name} @ threshold={thr:.2f}")

    bundle = {
        "model": model,
        "threshold": thr,
        "imputer": imp,
        "scaler": scaler,
        "pca": pca,
        "columns": list(df.drop('ckd', axis=1).columns)
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "best_model.joblib"))

if __name__ == "__main__":
    main()
