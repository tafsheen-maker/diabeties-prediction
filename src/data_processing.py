import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path='data/pima_diabetes.csv'):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'src/scaler.pkl')
    return X_train_scaled, X_test_scaled, y_train, y_test
