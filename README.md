📂 Project Overview

Diabetes is a chronic disease affecting millions globally. Early prediction helps reduce risks through timely diagnosis.
This project applies predictive analytics using machine learning models to estimate the probability of diabetes based on clinical features such as glucose level, BMI, insulin levels, and age.

The repository provides:
	•	Clean and reproducible code
	•	Notebooks for analysis
	•	Modular scripts for scaling, preprocessing, training, and evaluating models
	•	A Flask app for real-time predictions
	•	Clear folder structure for easy understanding

🎯 Objectives

✔ Understand the dataset through statistical summaries and visual insights
✔ Clean and preprocess medical health data
✔ Build predictive ML models
✔ Select the best-performing model
✔ Deploy the model using a simple Flask API
✔ Provide an interactive web form for real-time prediction

🧹 Data Preprocessing Steps

The raw data contains missing or zero values in medical features. Steps performed:

✔ Handling Missing Values
	•	Replaced zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI
	•	Imputed missing values using median imputation

✔ Feature Scaling
	•	Applied StandardScaler to normalize numeric features

✔ Train-Test Split
	•	Stratified split (80% train, 20% test)
	•	Ensures equal class distribution

✔ Optional Feature Engineering
	•	Added interaction features like:
	•	BMI_Age
	•	Preg_over_Age

📊 Exploratory Data Analysis

The EDA notebook includes:
	•	Distribution plots for each feature
	•	Correlation heatmap
	•	Outcome imbalance visualization
	•	Outlier analysis
	•	Pairplots to see relationship trends

These insights help in selecting relevant features and understanding the model behavior.

🤖 Machine Learning Models Used

Three models were trained and compared:

1️⃣ Logistic Regression

Simple baseline model
	•	Interpretable
	•	Good for linear relationships

2️⃣ Random Forest Classifier (Best Model)
	•	Handles non-linearity
	•	Robust to noise
	•	Performs well on medical datasets

3️⃣ XGBoost
	•	Gradient boosting algorithm
	•	Strong performance on tabular data

Evaluation metrics included:
	•	Classification report
	•	Confusion matrix
	•	ROC–AUC score
	•	Accuracy, Precision, Recall, F1-Score
  
  🔮 Future Improvements
	•	Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
	•	Implement cross-validation
	•	Add SMOTE to handle class imbalance
	•	Deploy on Render / Railway / AWS / Heroku
	•	Add Dockerfile for containerization
	•	Add CI/CD pipeline using GitHub Actions
