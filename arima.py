import pandas as pd
import numpy as np
import itertools
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import forecast


df = pd.read_csv("timeseries.csv")

# Convert 'datetime' to datetime type and set it as the index
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Retain the mean temperature column
df = df[['temp']]

# Plotting the data
fig = px.line(df.reset_index(), x='datetime', y='temp')
fig.update_layout(xaxis_title='Date', yaxis_title='Temperature')
# fig.show()

# Splitting the data into train and test sets
train_size = int(len(df) * 0.67)
train, test = df['temp'][:train_size], df['temp'][train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 2))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Calculate metrics
mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Predictions
forecast = model_fit.forecast(steps=10)

# Get future dates
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='D')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['temp'], label='Observed')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Temperature (°F)")
plt.title("Forecasted Temperature for Nairobi")
plt.grid(True)
# plt.show()

# Tuning the model
p = d = q = range(0,10)

# Generate Combinations
pdq = list(itertools.product(p, d, p))
results = pd.DataFrame(columns = ['pdq', 'MAE', 'MSE', 'RMSE'])

# Grid search for best parameters

for param in pdq:
    try:
        print(f"Trying ARIMA{param}")
        model = ARIMA(train, order=param)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        mae = mean_absolute_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        results = pd.concat([results, pd.DataFrame({'pdq': [param], 'MAE': [mae], 'MSE':[mse], 'RMSE':[rmse]})], ignore_index=True)

    except Exception as e:
        print(f"Exception for {param}: {e}")
        if hasattr(e, 'mle_retvals'):
            print(e.mle_retvals)

if results.empty:
    print("No results found.")
else:
    best_paramas = results.loc[results['MAE'].idxmin()]
    print("\nBest ARIMA Order:")
    print(f"  pdq: {best_paramas['pdq']}")
    print(f"  MAE: {best_paramas.MAE:.2f}")
    print(f"  MSE: {best_paramas.MSE:.2f}")
    print(f"  RMSE: {best_paramas.RMSE:.2f}")

# Forecast with best model

model = ARIMA(df['temp'], order=best_paramas['pdq'])
model_fit = model.fit()
predictions = model_fit.forecast(steps=10)

# Get future dates for final forecast
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='D')

# Plot the entire data + forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['temp'], label='Observed', color='blue')
plt.plot(forecast_dates, predictions, label='Final Forecast (Best ARIMA)', color='green', linestyle='--', marker='o')
plt.xlabel("Date")
plt.ylabel("Temperature (°F)")
plt.title("Final 10-Day Temperature Forecast for Nairobi (Using Best ARIMA Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

