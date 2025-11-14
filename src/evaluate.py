import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.data_processing import load_data, preprocess, split_and_scale

def evaluate_model(model_path):
    model = joblib.load(model_path)
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

if _name_ == '_main_':
    evaluate_model('src/model_rf.pkl')
