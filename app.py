

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide", page_title="PM2.5 Estimation Dashboard")

st.title("🌍 AI-Based Surface PM2.5 Prediction Dashboard")

# Sidebar Inputs
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload prediction_ready_grid.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load model
    model = joblib.load("random_forest_pm25_retrained.pkl")

    # Rename column if needed
    if "precipitation" in df.columns:
        df.rename(columns={"precipitation": "precipitation_sum"}, inplace=True)

    # Predict
    features = ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max", "precipitation_sum"]
    df["PM2.5_predicted"] = model.predict(df[features])

    # Display metrics
    st.subheader("📊 Model Input Features")
    st.write(df[features].head())

    st.subheader("📌 PM2.5 Predictions")
    st.write(df[["Latitude", "Longitude", "PM2.5_predicted"]].head())

    # Map plot
    st.subheader("🗺️ PM2.5 Prediction Map")
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["PM2.5_predicted"],
                    cmap="plasma", edgecolor="k", s=50)
    plt.colorbar(sc, label="PM2.5 (µg/m³)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    st.pyplot(fig)

    # Download CSV
    st.subheader("⬇️ Download Predictions")
    st.download_button("Download CSV", df.to_csv(index=False), "predicted_pm25.csv", "text/csv")

else:
    st.warning("Please upload the prediction-ready CSV file.")
