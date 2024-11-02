# Exponential Smoothing Forecasting

This project implements exponential smoothing techniques to forecast zonal prices and load data using the Holt-Winters method and naive models. The goal is to compare the accuracy of these forecasting models over different calibration periods.

## Project Overview

The project includes the following key components:

1. **Model Fitting**:
   - Reimplement a simple ETS (Error, Trend, Seasonality) model fitting process that iteratively evaluates a range of smoothing parameters (alpha) from 0.01 to 0.99, aiming to minimize the Mean Squared Error (MSE).
   - Utilize `scipy.optimize.minimize()` for fitting the ETS model in Python.

2. **Forecasting**:
   - Using GEFCOM zonal price data, compute Holt-Winters (HW) and three naive forecasting variants for one-step ahead predictions across all hours of the day (h = 1, 2, ..., 24) and for the average daily price. The first 360 days of data will be used to calibrate the HW model.
   - Evaluate and plot the Mean Absolute Error (MAE) and RMSE for each hour and average across all four models.

3. **Calibration Comparison**:
   - Repeat the forecasting tasks using the first 720 days of data for calibration of the HW model. Analyze whether a longer calibration window results in more or less accurate forecasts and focus on predictions for the same out-of-sample period (days #721, #722, ...).

4. **Zonal Load Forecasting**:
   - Extend the analysis to the zonal load data (column 5) using the same methodologies as applied to zonal prices.

## Data

The data used in this project consists of GEFCOM zonal price and load data, which can be obtained from the relevant sources. The analysis involves various columns from the dataset, particularly focusing on the zonal price (column 3) and zonal load (column 5).

## Results

The project outputs visualizations and performance metrics (MAE and RMSE) for the different forecasting models and calibration periods, facilitating a comparison of their accuracy. 
