from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(_name_)
# load model and scaler
model = joblib.load('../src/model_rf.pkl')  # or model_logistic.pkl
scaler = joblib.load('../src/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form or request.json
    # expected keys: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    try:
        vals = [float(data[k]) for k in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
    except Exception as e:
        return jsonify({'error': 'Invalid input format', 'details': str(e)}), 400
    X = np.array(vals).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0,1]
    pred = int(prob >= 0.5)
    return jsonify({'prediction': pred, 'probability': float(prob)})

if _name_ == '_main_':
    app.run(debug=True, host='0.0.0.0', port=5000)
