# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset (example: Apple stock prices from Yahoo Finance)
data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=0&period2=9999999999&interval=1d&events=history')

# Convert the 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check for missing values and fill or drop them as necessary
data = data.dropna()

# Plot the stock prices over time
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.title('Historical Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['Close'], model='additive', period=252) # Assuming 252 trading days in a year
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(data['Close'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Check for stationarity using the Augmented Dickey-Fuller test
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# If the series is not stationary, apply differencing
data_diff = data['Close'].diff().dropna()

# Plot the differenced series
plt.figure(figsize=(12, 6))
plt.plot(data_diff, label='Differenced Close Price')
plt.title('Differenced Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot ACF and PACF plots to determine AR and MA terms
plot_acf(data_diff, lags=20)
plt.show()
plot_pacf(data_diff, lags=20)
plt.show()

# Split the data into training and test sets
train_size = int(len(data_diff) * 0.8)
train, test = data_diff[:train_size], data_diff[train_size:]

# Fit an ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Forecast future stock prices
forecast = model_fit.forecast(steps=len(test))[0]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print('Test RMSE:', rmse)

# Plot the forecasted values along with the actual values
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, forecast, label='Forecast')
plt.title('ARIMA Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
