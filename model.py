import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_csv("students_mental_health_survey.csv")

df.replace({
    "Course": {'Engineering': 1, 'Business': 2, 'Medical': 3, 'Law': 4, "Computer Science": 5, 'Others': 6},
    "Sleep_Quality": {'Good': 3, 'Average': 2, 'Poor': 1},
    "Physical_Activity": {'High': 3, 'Moderate': 2, 'Low': 1},
    "Diet_Quality": {'Good': 3, 'Average': 2, 'Poor': 1},
    "Social_Support": {'High': 3, 'Moderate': 2, 'Low': 1},
    "Relationship_Status": {'Married': 1, 'Single': 2, 'In a Relationship': 3},
    "Substance_Use": {'Never': 1, 'Occasionally': 2, 'Frequently': 3},
    "Counseling_Service_Use": {'Never': 1, 'Occasionally': 2, 'Frequently': 3},
    "Family_History": {'Yes': 1, 'No': 2},
    "Chronic_Illness": {'Yes': 1, 'No': 2},
    "Extracurricular_Involvement": {'High': 1, 'Moderate': 2, 'Low': 3},
    "Residence_Type": {'With Family': 1, 'Off-Campus': 2, 'On-Campus': 3},
    "Gender": {'Male': 1, 'Female': 2}
}, inplace=True)

df.fillna(df.mode().iloc[0], inplace=True)
df.dropna(subset=["Depression_Score"], inplace=True)

X = df.drop(columns=["Depression_Score"])
y = df["Depression_Score"]

selector = SelectKBest(score_func=f_regression, k=18)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = X[selected_features]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

xgb = XGBRegressor(random_state=42)
xgb_random_search = RandomizedSearchCV(xgb, param_distributions=xgb_param_dist, n_iter=10, cv=3, verbose=2, n_jobs=1)
xgb_random_search.fit(X_train, y_train)
best_xgb = xgb_random_search.best_estimator_

y_pred = best_xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("XGBoost Model Performance")
print("MSE:", mse)
print("R² Score:", r2)

input_styles = {
    "Course": "1,2,3,4,5,6",
    "Sleep_Quality": "1,2,3",
    "Physical_Activity": "1,2,3",
    "Diet_Quality": "1,2,3",
    "Social_Support": "1,2,3",
    "Relationship_Status": "1,2,3",
    "Substance_Use": "1,2,3",
    "Counseling_Service_Use": "1,2,3",
    "Family_History": "1,2",
    "Chronic_Illness": "1,2",
    "Extracurricular_Involvement": "1,2,3",
    "Residence_Type": "1,2,3",
    "Gender": "1,2",
    "Age": "Continuous",
    "CGPA": "Continuous",
    "Financial_Stress": "1,2,3,4,5",
    "Semester_Credit_Load": "Continuous",
    "Anxiety_Score": "1,2,3,4,5",
    "Stress_Level": "1,2,3,4,5"
}

print(selected_features.tolist())
with open("stress_level_models.pkl", "wb") as f:
    pickle.dump({
        "xgboost": best_xgb,
        "scaler": scaler,
        "selected_features": selected_features.tolist(),
        "input_styles": input_styles
    }, f)
