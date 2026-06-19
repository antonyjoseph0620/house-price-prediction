import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("data/housing.csv")

# Feature Engineering
df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
df["HouseAge"] = 2026 - df["YearBuilt"]

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Numerical and Categorical Columns
numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "YearBuilt",
    "TotalRooms",
    "HouseAge"
]

categorical_features = ["Location"]

# Preprocessing for numerical data
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate model
score = r2_score(y_test, predictions)
print(f"R2 Score: {score:.4f}")

# Create model folder
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/house_price_model.pkl")

print("Model saved successfully!")