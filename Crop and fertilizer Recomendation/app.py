import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Custom CSS to enhance the UI
st.markdown("""
<style>
    .main {
        padding: 20px;
        max-height: 100vh;
        max-width: 90vh;
        margin: 0 auto;
        overflow-y: auto;
    }
    .stApp {
        height: 100vh;
        overflow: hidden;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-text {
        text-align: center;
        color: #2c3e50;
        padding: 15px;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle-text {
        text-align: center;
        color: #34495e;
        padding: 8px;
        font-size: 22px;
        margin: 20px;
    }
    .stSelectbox {
        margin-bottom: 10px;
    }
    .stNumberInput {
        margin-bottom: 10px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 80vh;
        margin: 0 auto;
    }
    .css-1d391kg, .css-12oz5g7 {
        padding: 1rem 1rem 1rem;
    }
    footer {
        position: fixed;
        bottom: 0;
        width: 80vh;
        margin: 0 auto;
        left: 0;
        right: 0;
        background: white;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved models and scalers
crop_model = pickle.load(open('model/crop_model.sav', 'rb'))
crop_scaler = pickle.load(open('model/crop_scaler.sav', 'rb'))
fertilizer_model = pickle.load(open('model/fertilizer_model.sav', 'rb'))
fertilizer_scaler = pickle.load(open('model/fertilizer_scaler.sav', 'rb'))

# crop dictionary 

crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
                 6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 
                 11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 
                 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",19: "Pigeonpeas", 
                 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# fertilizer dictionary 

fert_dict = {
'Urea':1,
'DAP':2,
'14-35-14':3,
'28-28':4,
'17-17-17':5,
'20-20':6,
'10-26-26':7,
}

soil_type_dict = {
    'Sandy': 1,
    'Loamy': 2,
    'Black': 3,
    'Red': 4,
    'Clayey': 5
}

crop_type_dict = {
    'Maize': 1,
    'Sugarcane': 2,
    'Cotton': 3,
    'Tobacco': 4,
    'Paddy': 5,
    'Barley': 6,
    'Wheat': 7,
    'Millets': 8,
    'Oil seeds': 9,
    'Pulses': 10,
    'Ground Nuts': 11
}

# Title and description
st.markdown('<p class="title-text">Smart Agriculture Assistant</p>', unsafe_allow_html=True)

# Create tabs for Crop and Fertilizer recommendation
tab1, tab2 = st.tabs(["Crop Recommendation", "Fertilizer Recommendation"])

with tab1:
    st.markdown('<p class="subtitle-text">Crop Recommendation System</p>', unsafe_allow_html=True)
    
    # Input fields for crop recommendation
    col1, col2 = st.columns(2)
    
    with col1:
        nitrogen = st.number_input('Nitrogen (N)', min_value=0, max_value=140)
        phosphorus = st.number_input('Phosphorus (P)', min_value=5, max_value=145)
        potassium = st.number_input('Potassium (K)', min_value=5, max_value=205)
        temperature = st.number_input('Temperature (°C)', min_value=8.0, max_value=44.0)
    
    with col2:
        humidity = st.number_input('Humidity (%)', min_value=14.0, max_value=100.0)
        ph = st.number_input('pH', min_value=3.5, max_value=10.0)
        rainfall = st.number_input('Rainfall (mm)', min_value=20.0, max_value=300.0)

    if st.button('Predict Crop'):
        # Prepare input data
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        # Scale the features
        scaled_features = crop_scaler.transform(features)
        # Make prediction
        prediction = crop_model.predict(scaled_features)
        
        st.success(f'The recommended crop for your soil conditions is: {crop_dict [prediction[0]]}')

with tab2:
    st.markdown('<p class="subtitle-text">Fertilizer Recommendation System</p>', unsafe_allow_html=True)
    
    # Input fields for fertilizer recommendation
    col1, col2 = st.columns(2)
    
    with col1:
        temperature_f = st.number_input('Temperature (°C)', min_value=25, max_value=38, key='temp_f')
        humidity_f = st.number_input('Humidity (%)', min_value=50, max_value=72, key='hum_f')
        moisture_f = st.number_input('Moisture', min_value=25, max_value=65)
        soil_type = st.selectbox('Soil Type', ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
    
    with col2:
        crop_type = st.selectbox('Crop Type', ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
                                               'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'])
        nitrogen_f = st.number_input('Nitrogen (N)', min_value=4, max_value=42, key='n_f')
        potassium_f = st.number_input('Potassium (K)', min_value=0, max_value=19, key='k_f')
        phosphorous_f = st.number_input('Phosphorous (P)', min_value=0, max_value=42, key='p_f')

    if st.button('Recommend Fertilizer'):
        # Create input array with all 8 numerical features
        input_data = np.array([[temperature_f, humidity_f, moisture_f, soil_type_dict[soil_type], 
                               crop_type_dict[crop_type], nitrogen_f, potassium_f, phosphorous_f]])
        
        # Scale the numerical features
        scaled_input = fertilizer_scaler.transform(input_data)
        
        # Make prediction
        fertilizer_prediction = fertilizer_model.predict(scaled_input)
        
        # Get the fertilizer name from the dictionary
        fertilizer_name = list(fert_dict.keys())[list(fert_dict.values()).index(fertilizer_prediction[0])]
        st.success(f'The recommended fertilizer is: {fertilizer_name}')


# Add footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Made with ❤️ for Smart Agriculture</p>
</div>
""", unsafe_allow_html=True)