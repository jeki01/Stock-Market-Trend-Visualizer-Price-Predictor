import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📈 Stock Market Trend Visualizer & Predictor")

# User input
ticker = st.text_input("Enter Stock Ticker Symbol (e.g. AAPL, MSFT, TSLA)", "AAPL")
days = st.slider("Select number of past days to visualize", min_value=10, max_value=100, value=30)

# Fetch Data
df = yf.download(ticker, period=f"{days+1}d")

if not df.empty:
    df = df.reset_index()
    df['Day'] = range(len(df))  # Convert date to number for ML model

    # Plot closing price
    st.subheader(f"📉 Closing Price for {ticker.upper()} (Last {days} Days)")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ML model
    model = LinearRegression()
    X = df[['Day']]
    y = df['Close']
    model.fit(X, y)
    next_day = np.array([[len(df)]])
    predicted_price = model.predict(next_day)[0]

    st.success(f"🧠 Predicted closing price for next day: **${predicted_price:.2f}**")

else:
    st.error("Could not fetch data. Please check the ticker symbol.")

