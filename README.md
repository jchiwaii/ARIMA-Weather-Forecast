# Nairobi Temperature Forecast (ARIMA)

This project uses an ARIMA model to forecast Nairobi's temperature for the next 10 days. Initially, a random ARIMA model is fitted and used to forecast values. A grid search then follows to identify the best parameters for `p`, `d`, and `q`.

The dataset covers one year: **from May 14, 2024, to May 14, 2025**.

Forecasts are made for the next **10 days**, and temperature is measured in **Fahrenheit**.

## Steps to Run

1. Install the required dependencies:

```bash
pip install -r requirements.txt
````

2. Ensure `timeseries.csv` is in the **same directory** as the Python script.

3. Run the script:

```bash
python arima.py
```

## Forecast Plot

View the interactive plots here:

- [Observed Temperature Plot](observed_temperature.html)
- [Initial Forecast Plot](initial_forecast.html)
- [Final Forecast Plot](final_forecast.html)


## Notes

* Developed and tested using **PyCharm**.
* Use a **virtual environment** for best experience.

