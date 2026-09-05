import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# =========================
# Load trained regression model
# =========================

model = tf.keras.models.load_model('regression_model.h5')


# =========================
# Load encoders and scaler
# =========================

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('regression_scaler.pkl', 'rb') as file:
    regression_scaler = pickle.load(file)


# =========================
# Streamlit App
# =========================

st.title("Estimated Salary Prediction")


# =========================
# Input fields
# =========================

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    18,
    92
)

balance = st.number_input(
    "Balance"
)

credit_score = st.number_input(
    "Credit Score"
)

tenure = st.slider(
    "Tenure",
    0,
    10
)

num_of_products = st.slider(
    "Number of Products",
    1,
    4
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# =========================
# Prepare input data
# =========================

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member]
})


# =========================
# One-hot encode Geography
# =========================

geography_encoded = onehot_encoder_geo.transform(
    pd.DataFrame({
        "Geography": [geography]
    })
).toarray()


geography_df = pd.DataFrame(
    geography_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(
        ["Geography"]
    )
)


# =========================
# Combine Geography with input
# =========================

input_data = pd.concat(
    [
        input_data.reset_index(drop=True),
        geography_df.reset_index(drop=True)
    ],
    axis=1
)


# =========================
# Arrange columns exactly
# as during training
# =========================

input_data = input_data[
    regression_scaler.feature_names_in_
]


# =========================
# Scale input
# =========================

input_data_scaled = regression_scaler.transform(
    input_data
)


# =========================
# Prediction
# =========================

prediction = model.predict(input_data_scaled)

predicted_salary = prediction[0][0]


# =========================
# Display result
# =========================

st.success(
    f"Predicted Estimated Salary: ₹{predicted_salary:,.2f}"
)