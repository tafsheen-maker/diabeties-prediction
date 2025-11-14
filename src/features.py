# Example: add useful features (BMI_age interaction)
def add_features(df):
    df = df.copy()
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['Preg_over_Age'] = df['Pregnancies'] / (df['Age'] + 1)
    return df
