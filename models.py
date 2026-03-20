
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def get_models():
    return {
        "logistic": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgb": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric='logloss', random_state=42),
        "nn": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400, random_state=42)
    }
