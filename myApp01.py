import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load the dataset for label encoding
df = pd.read_csv('C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/house_kg_10K_ads.csv')

# Label encoding for 'district', 'building_type', and 'condition' columns
labelencoder_X_1 = LabelEncoder()
df['district_encoded'] = labelencoder_X_1.fit_transform(df['district'])

labelencoder_X_2 = LabelEncoder()
df['building_type_encoded'] = labelencoder_X_2.fit_transform(df['building_type'])

labelencoder_X_3 = LabelEncoder()
df['condition_encoded'] = labelencoder_X_3.fit_transform(df['condition'])

# Define independent variables
independent_variables = ["square", "rooms", "floors", "floor", "date_year", 'district_encoded',
                         "building_type_encoded", "condition_encoded", "source", "year_bin"]

# Streamlit app
st.title("House Price Prediction")

# Input form for user data
st.write("Enter the details for house price prediction:")
square = st.number_input("Square Feet", min_value=0.0)
rooms = st.number_input("Number of Rooms", min_value=0)
floors = st.number_input("Number of Floors", min_value=0)
floor = st.number_input("Floor Number", min_value=0)
date_year = st.number_input("Year Built", min_value=0)
district = st.selectbox("District", df['district'].unique())
building_type = st.selectbox("Building Type", df['building_type'].unique())
condition = st.selectbox("Condition", df['condition'].unique())
source = st.selectbox("Source", df['source'].unique())
year_bin = st.selectbox("Year Bin", df['year_bin'].unique())

# Map the selected original names to their label-encoded values for prediction
district_encoded = labelencoder_X_1.transform([district])[0]
building_type_encoded = labelencoder_X_2.transform([building_type])[0]
condition_encoded = labelencoder_X_3.transform([condition])[0]

# Model selection
model_name = st.selectbox("Choose a Model", ["DecisionTreeRegressor", "LinearRegression", "RandomForestRegressor"])

# Load the selected model
if model_name == "DecisionTreeRegressor":
    with open('DecisionTreeRegressor.pkl', 'rb') as file:
        model = pickle.load(file)
elif model_name == "LinearRegression":
    with open('LinearRegression.pkl', 'rb') as file:
        model = pickle.load(file)
elif model_name == "RandomForestRegressor":
    with open('RandomForestRegressor.pkl', 'rb') as file:
        model = pickle.load(file)

if st.button("Predict Price"):
    # Create a DataFrame with user inputs and label-encoded values
    user_data = pd.DataFrame({
        'square': [square],
        'rooms': [rooms],
        'floors': [floors],
        'floor': [floor],
        'date_year': [date_year],
        'district_encoded': [district_encoded],
        'building_type_encoded': [building_type_encoded],
        'condition_encoded': [condition_encoded],
        'source': [source],
        'year_bin': [year_bin]
    })

    # Standardize the user input data
    user_data[independent_variables] = (user_data[independent_variables] - df[independent_variables].mean()) / (
                df[independent_variables].std())

    # Make predictions using the selected model
    predicted_price = model.predict(user_data)

    st.success(f"Predicted Price: {predicted_price[0]:.2f} KGS")
