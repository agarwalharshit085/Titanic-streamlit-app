# Date :- 11 - 04 - 2025
# Name :- Harshit Agrawal

# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("titanic.csv")
    return data

df = load_data()

st.title("🚢 Titanic Survival Prediction")

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {acc:.2f}")

st.sidebar.header("Enter Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 5, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode inputs
sex_encoded = 1 if sex == 'male' else 0
embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Predict for user input
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
prediction = model.predict(input_data)

if st.sidebar.button("Predict"):
    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to **Survive**.")
    else:
        st.error("❌ The passenger is likely to **Not Survive**.")

st.write("#### Sample of Cleaned Dataset:")
st.dataframe(df.head())
