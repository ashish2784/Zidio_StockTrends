```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
from datetime import datetime

# Set plot style
sns.set(style='whitegrid')

# Model Definitions (same as notebook)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Data Collection and Preprocessing
@st.cache_data  # Cache data to avoid repeated downloads
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    ts = data['Close'].dropna()
    return ts

# Model Training and Forecasting Functions
def train_arima(ts, train_size):
    train, test = ts[:train_size], ts[train_size:]
    model = ARIMA(train, order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=len(test))
    return forecast, mean_squared_error(test, forecast)

def train_sarima(ts, train_size):
    train, test = ts[:train_size], ts[train_size:]
    model = SARIMAX(train, order=(4, 1, 0), seasonal_order=(1, 1, 1, 5))
    fit = model.fit()
    forecast = fit.forecast(steps=len(test))
    return forecast, mean_squared_error(test, forecast)

def train_prophet(ts, train_size):
    df = pd.DataFrame({'ds': ts.index, 'y': ts.values})
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    pred = forecast['yhat'].iloc[-len(test):]
    return pred, mean_squared_error(test['y'], pred)

def train_lstm(ts, train_size, seq_length=5):
    scaler = MinMaxScaler()
    scaled_ts = scaler.fit_transform(ts.values.reshape(-1, 1))
    
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    X, y = create_sequences(scaled_ts, seq_length)
    train_X, test_X = X[:train_size-seq_length], X[train_size-seq_length:]
    train_y, test_y = y[:train_size-seq_length], y[train_size-seq_length:]
    
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
    
    lstm_predictions = []
    current_seq = test_X[0].unsqueeze(0)
    for _ in range(len(test_y)):
        model.eval()
        with torch.no_grad():
            pred = model(current_seq)
        lstm_predictions.append(pred.item())
        current_seq = torch.cat((current_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)), dim=1)
    
    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
    return lstm_predictions.flatten(), mean_squared_error(test_y.numpy(), lstm_predictions)

# Streamlit App
st.title("Stock Market Forecasting Dashboard - ASIANPAINT.NS")

# Sidebar for inputs
st.sidebar.header("Configure Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", "ASIANPAINT.NS")
start_date = st.sidebar.date_input("Start Date", datetime(2000, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())
train_size = st.sidebar.slider("Train Size (% of data)", 50, 90, 80)

# Load data
ts = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
train_size = int(len(ts) * train_size / 100)

# Plot Original Data
st.subheader("Historical Close Price")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(ts, label='Close Price')
ax1.set_title('Historical Close Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)

# Train and Display Forecasts
if st.button("Generate Forecasts"):
    st.subheader("Model Forecasts")
    
    # ARIMA
    arima_forecast, arima_mse = train_arima(ts, train_size)
    st.write(f"ARIMA MSE: {arima_mse}")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(ts.index[:train_size], ts[:train_size], label='Train')
    ax2.plot(ts.index[train_size:], ts[train_size:], label='Test')
    ax2.plot(ts.index[train_size:], arima_forecast, label='ARIMA Forecast')
    ax2.legend()
    st.pyplot(fig2)
    
    # SARIMA
    sarima_forecast, sarima_mse = train_sarima(ts, train_size)
    st.write(f"SARIMA MSE: {sarima_mse}")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(ts.index[:train_size], ts[:train_size], label='Train')
    ax3.plot(ts.index[train_size:], ts[train_size:], label='Test')
    ax3.plot(ts.index[train_size:], sarima_forecast, label='SARIMA Forecast')
    ax3.legend()
    st.pyplot(fig3)
    
    # Prophet
    prophet_forecast, prophet_mse = train_prophet(ts, train_size)
    st.write(f"Prophet MSE: {prophet_mse}")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(ts.index[:train_size], ts[:train_size], label='Train')
    ax4.plot(ts.index[train_size:], ts[train_size:], label='Test')
    ax4.plot(ts.index[train_size:], prophet_forecast, label='Prophet Forecast')
    ax4.legend()
    st.pyplot(fig4)
    
    # LSTM
    lstm_forecast, lstm_mse = train_lstm(ts, train_size)
    st.write(f"LSTM MSE: {lstm_mse}")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(ts.index[train_size+5:], ts[train_size+5:], label='Test')
    ax5.plot(ts.index[train_size+5:train_size+5+len(lstm_forecast)], lstm_forecast, label='LSTM Forecast')
    ax5.legend()
    st.pyplot(fig5)

    # Model Comparison
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
        'MSE': [arima_mse, sarima_mse, prophet_mse, lstm_mse]
    })
    st.table(comparison)
    st.write(f"Best Model: {comparison.loc[comparison['MSE'].idxmin()]['Model']} with MSE: {comparison['MSE'].min()}")

# Run the app
if __name__ == "__main__":
    st.sidebar.write("Last Updated: 02:48 PM IST, September 30, 2025")
```
