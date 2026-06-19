# house-price-prediction
# 🏠 House Price Prediction using Machine Learning

## Overview

This project is a Machine Learning-based House Price Prediction system that estimates the price of a house based on various features such as area, number of bedrooms, number of bathrooms, location, and year built.

The model is trained using a Random Forest Regressor and deployed through an interactive Streamlit web application, allowing users to enter property details and instantly receive an estimated house price.

## Features

* Predict house prices based on property characteristics.
* Feature engineering using Total Rooms and House Age.
* Data preprocessing with Scikit-learn Pipelines.
* Interactive web interface built with Streamlit.
* Model persistence using Joblib.
* Git and GitHub integration for version control.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Git & GitHub

## Dataset Features

| Feature    | Description                     |
| ---------- | ------------------------------- |
| Area       | House area in square feet       |
| Bedrooms   | Number of bedrooms              |
| Bathrooms  | Number of bathrooms             |
| Location   | City where the house is located |
| YearBuilt  | Year the house was built        |
| TotalRooms | Bedrooms + Bathrooms            |
| HouseAge   | Current Year - Year Built       |
| Price      | Target variable (House Price)   |

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Train-Test Split
5. Model Training using Random Forest Regressor
6. Model Evaluation using R² Score
7. Model Saving using Joblib
8. Deployment using Streamlit

## Project Structure

house-price-prediction/
│
├── data/
│   └── housing.csv
│
├── model/
│   └── house_price_model.pkl
│
├── app.py
├── train.py
├── README.md
└── .gitignore

## Installation

Clone the repository:

git clone https://github.com/antonyjoseph0620/house-price-prediction.git

Move into the project folder:

cd house-price-prediction

Install dependencies:

pip install -r requirements.txt

## Run the Application

Train the model:

python train.py

Run the Streamlit app:

streamlit run app.py

## Model Performance

* Algorithm: Random Forest Regressor
* Evaluation Metric: R² Score
* Achieved Accuracy (R²): Approximately 0.98

## Future Enhancements

* Add more real-world housing data.
* Implement advanced feature engineering.
* Deploy the application on Streamlit Cloud.
* Add visual analytics and price trend charts.
* Integrate additional ML algorithms for comparison.

## Author

Antony Joseph

GitHub: https://github.com/antonyjoseph0620
