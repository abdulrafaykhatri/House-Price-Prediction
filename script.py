import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Function to load the pickled model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model   

# Function to make predictions using the loaded model
def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction

df = pd.read_csv('./Data/Entities.csv')
location_frequency_map = df['location'].value_counts(normalize=True)
category_mapping = {'Room': 0, 'Lower Portion': 1, 'Upper Portion': 2, 'Flat': 3, 'House': 4, 'Penthouse': 5, 'Farm House': 6}
unique_property_types = df['property_type'].unique()
unique_locations = df['location'].unique()
unique_cities = df['city'].unique()
unique_purpose = df['purpose'].unique()

# Load the pickled model
model_path = 'predictive_model.pkl'
model = load_model(model_path)

def main():
    st.title("House Price Prediction Model")
    st.subheader("Enter house details for prediction")

    # Load data for normalization and mapping
    
    
    property_type = st.selectbox("Property Type", options=unique_property_types)
    city = st.selectbox("City", unique_cities)
    location = st.selectbox("Location", options=unique_locations)
    baths = st.number_input("Bath Rooms", step=1, min_value=0)
    if baths % 1 != 0:
        st.warning("Please enter an integer value.")
    bedrooms = st.number_input("Bed Rooms", step=1, min_value=0)
    if bedrooms % 1 != 0:
        st.warning("Please enter an integer value.")
    total_area = st.number_input("Total Area ")
    purpose = st.selectbox("Purpose", options=unique_purpose)
    
    if st.button("Predict", key="predict_button"):
        input_data = [
            category_mapping.get(property_type, 0),
            location_frequency_map.get(location[0], 0),
            baths,
            bedrooms,
            total_area,
            1 if city == 'Faisalabad' else 0,
            1 if city == "Islamabad" else 0,
            1 if city == "Karachi" else 0,
            1 if city == "Lahore" else 0,
            1 if city == "Rawalpindi" else 0,
            1 if purpose == "For Rent" else 0,
            1 if purpose == "For Sale" else 0,
        ]
        prediction = make_prediction(model, np.array(input_data).reshape(1, -1))
        st.success(f"Predicted Price: Rs. {prediction[0]:,.2f}")

if __name__ == '__main__':
    main()