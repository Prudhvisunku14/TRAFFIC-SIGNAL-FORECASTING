import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.ndimage import uniform_filter1d

# 📌 Define the model class (must match the one used during training)
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.trend_decoder = nn.Linear(d_model, 1)
        self.seasonal_decoder = nn.Linear(d_model, 1)
        self.pos_encoder = nn.Parameter(torch.randn(5000, d_model), requires_grad=False)

    def forward(self, src):
        x = self.input_proj(src)
        x = x + self.pos_encoder[:x.size(1)]
        x = self.transformer(x)
        out = self.decoder(x).squeeze(-1)
        trend = self.trend_decoder(x).squeeze(-1)
        seasonal = self.seasonal_decoder(x).squeeze(-1)
        return out, trend, seasonal

# 🚀 Load the model
model = TimeSeriesTransformer()
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

# 🎛️ Streamlit App UI
st.set_page_config(page_title="Traffic Forecasting App", layout="centered")
st.markdown("<h1 style='text-align: center;'>🚦 Traffic Forecasting App</h1>", unsafe_allow_html=True)

# User inputs
option = st.selectbox("Select forecast duration:", ["1 Day", "1 Week", "1 Month"])
date_str = st.text_input("Enter forecast start date (YYYY-MM-DD):", "2025-06-01")
junction_num = st.slider("Enter Junction Number (1 to 4):", 1, 4, 1)

# Mapping durations
if option == "1 Day":
    forecast_hours = 24
    input_hours = 96
elif option == "1 Week":
    forecast_hours = 168
    input_hours = 672
elif option == "1 Month":
    forecast_hours = 720
    input_hours = 2880

# 🧠 Run Forecast
if st.button("Run Forecast"):
    try:
        input_tensor = torch.randn(1, input_hours, 1)  # Replace this with real input later

        with torch.no_grad():
            out, trend_out, seasonal_out = model(input_tensor)

        prediction = out[0, -forecast_hours:].numpy()
        trend = trend_out[0, -forecast_hours:].numpy()
        seasonality = seasonal_out[0, -forecast_hours:].numpy()

        forecast_start = pd.Timestamp(date_str + " 00:00:00")
        forecast_times = [forecast_start + timedelta(hours=i) for i in range(forecast_hours)]

        # Confidence interval
        residual_std = np.std(prediction - trend)
        upper = prediction + 1.96 * residual_std
        lower = prediction - 1.96 * residual_std

        # 📈 Plotting
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(forecast_times, prediction, color='red', label='Forecast')
        ax.plot(forecast_times, trend, color='green', label='Trend')
        ax.plot(forecast_times, seasonality, color='blue', label='Seasonality')
        ax.fill_between(forecast_times, lower, upper, color='blue', alpha=0.2, label='95% CI')
        ax.axvline(forecast_times[0], color='gray', linestyle='--')
        ax.set_xlabel("Time")
        ax.set_ylabel("Traffic")
        ax.set_title(f"Forecast for Junction {junction_num} ({option})")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
