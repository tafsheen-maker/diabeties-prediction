# Model Report

## Dataset
Pima Indians Diabetes Dataset — 768 samples, 8 features + target.

## Models trained
- Logistic Regression
- Random Forest (n=100)
- XGBoost

## Best model (example)
Random Forest had the highest AUC (example value: 0.85). See notebooks for full metrics.

## Recommendations
- Add cross-validation and hyperparameter tuning (GridSearchCV or RandomizedSearchCV).
- Consider SMOTE if class imbalance affects recall.
- For production, create a separate data validation pipeline.
