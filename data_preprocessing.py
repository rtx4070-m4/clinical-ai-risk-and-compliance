
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess(df: pd.DataFrame):
    df = df.copy()
    # map categoricals
    df['rbc'] = df['rbc'].map({'normal':0,'abnormal':1})
    df['hypertension'] = df['hypertension'].map({'no':0,'yes':1})
    # separate
    X = df.drop('ckd', axis=1)
    y = df['ckd'].values
    # impute
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    # scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns)
    return X_scaled, y, imp, scaler
