import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------------
# Load model and data
# -------------------------

model = load_model("lstm_model.h5", compile=False)

scaler = joblib.load("scaler.pkl")

df = joblib.load("history_df.pkl")

future_df = joblib.load("future_df.pkl")

# -------------------------
# Page UI
# -------------------------

st.set_page_config(page_title="Climate Prediction", layout="wide")

st.title("🌍 Climate Change Analysis & Prediction")

st.markdown("""
This application predicts **future global land temperature**
using **LSTM Deep Learning model** trained on historical data (1750–2015).
""")

# -------------------------
# Input selection
# -------------------------

option = st.radio(
    "Choose prediction type",
    ["Predict by Date", "Predict by Year"]
)

# -------------------------
# Functions
# -------------------------

def get_temp_date(date_str):

    date = pd.to_datetime(date_str)

    # historical value
    if date in df.index:
        temp = df.loc[date, "LandAverageTemperature"]
        return f"Actual Temperature on {date.date()} : {temp:.2f} °C"

    # predicted value
    elif date in future_df.index:
        temp = future_df.loc[date, "PredictedTemp"]
        return f"Predicted Temperature on {date.date()} : {temp:.2f} °C"

    else:
        return "Date not available"


def get_temp_year(year):

    # historical avg
    if year in df.index.year:

        data = df[df.index.year == year]

        avg_temp = data["LandAverageTemperature"].mean()

        return f"Actual Average Temperature for {year} : {avg_temp:.2f} °C"

    # predicted avg
    elif year in future_df.index.year:

        data = future_df[future_df.index.year == year]

        avg_temp = data["PredictedTemp"].mean()

        return f"Predicted Average Temperature for {year} : {avg_temp:.2f} °C"

    else:

        return "Year not available"


# -------------------------
# UI controls
# -------------------------

if option == "Predict by Date":

    date_input = st.date_input(
        "Select date",
        value=pd.to_datetime("2030-01-01")
    )

    if st.button("Predict temperature"):

        result = get_temp_date(str(date_input))

        st.success(result)


if option == "Predict by Year":

    year_input = st.slider(
        "Select year",
        1750,
        2100,
        2030
    )

    if st.button("Predict average temperature"):

        result = get_temp_year(year_input)

        st.success(result)


# -------------------------
# Graph
# -------------------------

st.subheader("📈 Global Temperature Trend (1750–Future)")

combined = pd.concat([
    df.rename(columns={"LandAverageTemperature": "Temp"}),
    future_df.rename(columns={"PredictedTemp": "Temp"})
])

st.line_chart(combined["Temp"])


# -------------------------
# Model info
# -------------------------

st.subheader("🤖 Model Information")

st.write("""
Model : LSTM (Long Short-Term Memory)

Dataset : Global Land Temperature (1750–2015)

Framework : TensorFlow / Keras

Input : Past 12 months temperature

Output : Future temperature prediction
""")