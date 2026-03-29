import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans



class KMeansCluster(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X[["latitude", "longitude"]])
        return self

    def transform(self, X):
        X = X.copy()
        X["location_cluster"] = self.kmeans_.predict(X[["latitude", "longitude"]]).astype(str)
        return X

@st.cache_resource
def load_model():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "final_model.joblib")
    return joblib.load(model_path)


try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

model = load_model()

# ─── Title ─────────────────────────────────
st.title("🏠 California House Price Predictor")

st.write("Enter the details below to predict house price")

# ─── Inputs ────────────────────────────────
st.header("📍 Location")
latitude = st.number_input("Latitude", 32.0, 42.0, 37.77)
longitude = st.number_input("Longitude", -124.0, -114.0, -122.42)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "INLAND", "<1H OCEAN", "NEAR OCEAN", "ISLAND"]
)

st.header("🏠 Property Details")
housing_median_age = st.number_input("Housing Age", 1, 52, 28)
total_rooms = st.number_input("Total Rooms", 1, 5000, 2000)
total_bedrooms = st.number_input("Total Bedrooms", 1, 1500, 400)
households = st.number_input("Households", 1, 2000, 380)

st.header("👥 Demographics")
population = st.number_input("Population", 1, 15000, 1200)
median_income = st.number_input("Median Income", 0.5, 15.0, 4.5)

# ─── Derived Features ──────────────────────
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

# ─── Predict ───────────────────────────────
if st.button("Predict Price"):

    input_dict = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]

    st.success(f"🏡 Predicted Price: ${prediction:,.0f}")