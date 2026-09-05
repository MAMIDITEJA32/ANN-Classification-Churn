import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load the trained model
model = tf.keras.models.load_model('model.h5')


# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit app
st.title("Customer Churn Prediction")


# Input fields
geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider(
    'Age',
    18,
    92
)

balance = st.number_input(
    'Balance'
)

credit_score = st.number_input(
    'CreditScore'
)

estimated_salary = st.number_input(
    'Estimated salary'
)

tenure = st.slider(
    'Tenure',
    0,
    10
)

num_of_products = st.slider(
    'Number of Products',
    1,
    4
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)


# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One-hot encode Geography
geography_encoded = onehot_encoder_geo.transform(
    pd.DataFrame({'Geography': [geography]})
).toarray()


# Convert encoded Geography into DataFrame
geography_df = pd.DataFrame(
    geography_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)


# Combine original input data with Geography columns
input_data = pd.concat(
    [input_data.reset_index(drop=True), geography_df],
    axis=1
)


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn probability
prediction = model.predict(input_data_scaled)

prediction_probability = prediction[0][0]


# Display result
if prediction_probability > 0.5:

    st.error(
        f"The customer is likely to churn "
        f"with a probability of {prediction_probability:.2f}"
    )

else:

    st.success(
        f"The customer is unlikely to churn "
        f"with a probability of {prediction_probability:.2f}"
    )