"""
Script to train models. Run in a Python environment.
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.data_processing import load_data, preprocess, split_and_scale

def train_and_save():
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    # logistic
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    joblib.dump(log, 'src/model_logistic.pkl')
    # random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'src/model_rf.pkl')
    # xgboost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'src/model_xgb.pkl')

if _name_ == '_main_':
    train_and_save()
    print("Models trained and saved in src/")
