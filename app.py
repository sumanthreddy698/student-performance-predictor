
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("📘 Student Performance Predictor")
st.write("Predict student marks based on study hours, attendance & previous score.")

data = pd.read_csv("student_performance.csv")

X = data[['hours_studied','attendance','past_scores','extra_classes']]
y = data['performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

st.sidebar.header("Enter Student Details")
hours = st.sidebar.slider("Hours Studied",1,12,6)
attendance = st.sidebar.slider("Attendance %",40,100,75)
past_score = st.sidebar.slider("Previous Score",30,100,65)
extra = st.sidebar.radio("Extra Coaching?",[0,1])

input_df = pd.DataFrame([[hours,attendance,past_score,extra]],
                        columns=['hours_studied','attendance','past_scores','extra_classes'])

prediction = model.predict(input_df)[0]
prediction = max(0,min(prediction,100))

st.subheader("🎯 Predicted Student Performance")
st.success(f"Expected Score: {prediction:.2f} / 100")

y_pred=model.predict(X_test)
st.write("### 📊 Model Evaluation")
st.write(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
st.write(f"R² Score: {r2_score(y_test,y_pred):.2f}")
