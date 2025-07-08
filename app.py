import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# App Title
st.title("📈 Stock Market Trend Visualizer & Price Predictor")

# User input for stock ticker and number of days
ticker = st.text_input("Enter Stock Ticker Symbol (e.g. AAPL, MSFT, TSLA)", "AAPL")
days = st.slider("Select number of past days to visualize", min_value=10, max_value=100, value=30)

# Fetch historical data using yfinance
df = yf.download(ticker, period=f"{days+1}d")

# If data is fetched successfully
if not df.empty:
    df = df.reset_index()
    df['Day'] = range(len(df))  # Create numerical index for ML

    # Plot closing price trend
    st.subheader(f"📉 Closing Price Trend for {ticker.upper()} (Last {days} Days)")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], marker='o', color='blue', label='Closing Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Stock Price Trend")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Train Linear Regression model
    X = df[['Day']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)

    # Predict next day's closing price
    next_day = np.array([[len(df)]])
    predicted_price = model.predict(next_day)

    # Show prediction
    st.success(f"🧠 Predicted closing price for next day: **${predicted_price[0]:.2f}**")

else:
    st.error("❌ Could not fetch data. Please check the ticker symbol and try again.")
