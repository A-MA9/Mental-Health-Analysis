# Mental Health Analysis System

## Overview

This project is a **machine learning-based system** that predicts a student's **mental health risk level** based on multiple mental health and lifestyle factors. The **Streamlit-based web application** takes user inputs and provides predictions using a pre-trained **XGBoost regression model**.

## Features

- **User-friendly Streamlit interface** for input collection.
- **Machine Learning model (XGBoost)** trained on student mental health data.
- **Categorical inputs** (Course, Sleep Quality, etc.) are automatically encoded.
- **Numerical inputs** (CGPA, Age, Anxiety Score, etc.) are standardized before prediction.
- **Mental health risk classification:**
  - **Low Risk:** Score < 2 → Healthy mental state ✅
  - **Moderate Risk:** Score 2.1 - 3.5 → Some signs of distress ⚠️
  - **High Risk:** Score > 3.5 → High risk, seek help 🚨

## Dataset
This project uses the **Student Mental Health Assessment** dataset from [NidhiU-24's GitHub repository](https://github.com/NidhiU-24/Student-Mental-Health-Assessment/blob/main/students_mental_health_survey.csv).

## Architecture

1. **Data Preprocessing:**
   - Feature selection using `SelectKBest`
   - Standardization using `StandardScaler`
   - Categorical values converted to numeric representations
2. **Model Training:**
   - Uses `XGBoostRegressor`
   - Hyperparameter tuning via `RandomizedSearchCV`
   - Evaluated using Mean Squared Error (MSE) and R² Score
3. **Deployment:**
   - Model stored in **AWS S3**
   - Streamlit app running on **AWS EC2**
   - Fetches model from S3 and runs predictions in real-time

## Installation & Setup

### Prerequisites

- Python 3.7+
- AWS account (for EC2 and S3 setup)
- Libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`

### Steps

1. Clone this repository:
   ```bash
   git clone [https://github.com/your-repo/depression-prediction.git](https://github.com/A-MA9/Mental-Health-Analysis.git)
   cd depression-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the web interface.
2. Select categorical values from dropdowns.
3. Enter numerical values as required.
4. Click **Analyze Mental Health** to get an instant risk assessment.

## Future Improvements

- Adding **more features** to improve accuracy.
- Deploying a **real-time chatbot** for mental health support.
- Integrating **cloud-based database** to track student mental health trends.

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. For major changes, please open an issue first.

## License

This project is licensed under the MIT License.

---

Let me know if you'd like to modify or add anything! 🚀

