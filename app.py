import streamlit as st
import joblib
import pandas as pd
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "model",
    "house_price_model.pkl"
)

model = joblib.load(model_path)

st.title("🏠 House Price Prediction")

area = st.number_input("Area (sq ft)", min_value=500)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

location = st.selectbox(
    "Location",
    ["Chennai", "Coimbatore", "Trichy"]
)

year_built = st.number_input(
    "Year Built",
    min_value=1990,
    max_value=2026,
    value=2020
)

if st.button("Predict Price"):

    house_age = 2026 - year_built
    total_rooms = bedrooms + bathrooms

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Location": [location],
        "YearBuilt": [year_built],
        "TotalRooms": [total_rooms],
        "HouseAge": [house_age]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Price: ₹{prediction[0]:,.0f}"
    )