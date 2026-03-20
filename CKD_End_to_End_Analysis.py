
# Run this as a script or convert to notebook in Colab

!pip install -q xgboost shap lime seaborn joblib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess
from src.models import get_models
from src.evaluation import evaluate

df = pd.read_csv('data/raw/ckd_synthetic.csv')
X, y, _, _ = preprocess(df)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = get_models()
for name, m in models.items():
    m.fit(X_tr, y_tr)
    metrics = evaluate(m, X_te, y_te, threshold=0.5)
    print(name, metrics["recall"], metrics["fn"])
