import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore, storage
import uuid




# CSS styles
st.markdown(
    """
    <style>
        /* Main font for the entire container */
        .reportview-container {
            font-family: 'Arial', sans-serif;
        }
        /* Streamlit's default white background for the main content area */
        .main .block-container {
            background-color: #f7f7f8;
            padding-top: 5rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 5rem;
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #293846;
            color: white;
        }
        .sidebar h2 {
            color: white;
        }
        .Widget>label {
            color: #898989;
            font-weight: 400;
        }
        /* Button styling */
        .stButton>button {
            font-size: 1em;
            border-radius: 0.3em;
            border: 1px solid #2e7bcf;
            transition: background-color 0.3s, transform 0.3s;
        }
        .stButton>button:focus {
            outline: none;
            border-color: #4da3ff;
            box-shadow: 0 0 0.5em rgba(77,163,255,0.5);
        }
        .stButton>button:hover {
            background-color: #4da3ff;
            transform: translateY(-3px);
            box-shadow: none;
        }
        /* Text input styling */
        .stTextInput>div>div>input {
            color: #4f4f4f;
            border-radius: 0.3em;
            border: 1px solid #ced4da;
        }
        .stTextInput>div>div>input:focus {
            outline: none;
            border-color: #80bdff;
            box-shadow: 0 0 0.25em rgba(128,189,255,.25);
        }
        /* Dataframe styling */
        .stDataFrame {
            background-color: white;
            border-radius: 0.3em;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Load each model individually
with open('DecisionTreeRegressor.pkl', 'rb') as file1:
    decision_tree_model = pickle.load(file1)

with open('RandomForestRegressor_model.pkl', 'rb') as file9:
    random_forest_model = pickle.load(file9)
    
        
# Load label encoders
with open('district_encoder.pkl', 'rb') as file12:
    district_encoder = pickle.load(file12)
with open('micro_district_encoder.pkl', 'rb') as file13:
    micro_district_encoder = pickle.load(file13)
with open('building_type_encoder.pkl', 'rb') as file14:
    building_type_encoder = pickle.load(file14)
with open('condition_encoder.pkl', 'rb') as file15:
    condition_encoder = pickle.load(file15)    

# Streamlit interface
st.title("House Price Prediction")


# Sidebar - Page selection and model selection
st.sidebar.header('House Prices Prediction ')
page_options = ['Analyze Prices', 'Predict Prices']
page = st.sidebar.selectbox("Which page do you want to visit?", page_options)

# Load the data
house_data = pd.read_csv('C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/house_kg_10K_ads.csv')

