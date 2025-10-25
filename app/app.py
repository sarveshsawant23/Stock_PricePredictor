import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")

# Load trained model
model = joblib.load('model/stock_predictor.pkl')
features = ['SMA_10', 'SMA_50', 'RSI']

# Function to fetch and prepare data
def prepare_data(ticker):
    df = yf.download(ticker, start="2018-01-01", end="2025-01-01")
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df

# UI
st.title("📈 Stock Price Direction Predictor")
st.write("Predict whether a stock price will go **UP** or **DOWN** tomorrow.")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")

if ticker:
    try:
        df = prepare_data(ticker)
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        prediction = model.predict(X_scaled[-1].reshape(1, -1))

        if prediction == 1:
            st.success(f"The model predicts that {ticker} will go UP tomorrow 📈")
        else:
            st.error(f"The model predicts that {ticker} will go DOWN tomorrow 📉")

        # Optional: show last 100 days closing price
        st.line_chart(df['Close'].tail(100))

    except Exception as e:
        st.warning(f"Error fetching data for {ticker}. Please check the ticker symbol.")
