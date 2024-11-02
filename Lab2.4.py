import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin

# Load GEFCom2014 data
# (6 columns: YYYYMMDD, HH, zonal price, system load, zonal load, day-of-the-week)
d = np.loadtxt('GEFCOM.txt')

# Convert data to a Pandas DataFrame
df = pd.DataFrame(d, columns=['YYYYMMDD', 'HH', 'Zonal price', 'System load', 'Zonal load', 'Day of the week'])

# Create empty lists to store MAE and RMSE for all models
mae_hw = []
rmse_hw = []
mae_naive1 = []
rmse_naive1 = []
mae_naive2 = []
rmse_naive2 = []
mae_naive3 = []
rmse_naive3 = []

# Last day of the calibration period
T = 360
T1 = 720 #longer calibration window

# Period (weekly, i.e., 7-day, for daily data)
s = 7

# Holt-Winters parameters
initial_param = np.array([0.5, 0.5, 0.5])

# Define functions for Holt-Winters and forecasts
def holtwinters(param, s, data):
    alpha, beta, gamma = param
    y = data.copy()
    seasonals = np.zeros(len(data))

    for t in range(s, len(data)):
        y[t] = alpha * (data[t] - seasonals[t - s]) + (1 - alpha) * y[t - 1]
        seasonals[t] = gamma * (data[t] - y[t]) + (1 - gamma) * seasonals[t - s]

    mae = np.mean(np.abs(data[T + 1:] - y[T + 1:]))
    return mae

def holtwinters_forecast(param, s, data):
    alpha, beta, gamma = param
    y = data.copy()
    seasonals = np.zeros(len(data))

    for t in range(s, len(data)):
        y[t] = alpha * (data[t] - seasonals[t - s]) + (1 - alpha) * y[t - 1]
        seasonals[t] = gamma * (data[t] - y[t]) + (1 - gamma) * seasonals[t - s]

    return y

# Function to calculate MAE and RMSE for a specific model
def calculate_mae_rmse(data, forecast):
    mae = np.mean(np.abs(data - forecast))
    rmse = np.sqrt(np.mean((data - forecast) ** 2))
    return mae, rmse

for h in range(1, 25):
    # Filter data for the current hour
    p_hour = df[df['HH'] == h]['Zonal price'].values
    daily_price = df['Zonal price'].values

    # Holt-Winters
    param = fmin(holtwinters, initial_param, args=(s, p_hour[0:T]))
    pf_hw = holtwinters_forecast(param, s, p_hour[T:])
    mae_hw.append(np.mean(np.abs(p_hour[T1:] - pf_hw[T:])))
    rmse_hw.append(np.sqrt(np.mean((p_hour[T1:] - pf_hw[T:]) ** 2)))

    #co się tu właściwie zmienia? dla HW prognoza jest liczona dla T, ale mae od T1
    #czy dla Naivów się cokolwiek zmienia??

    # 1st Naive Forecast
    pf_naive1 = df['Zonal price'].shift(7 * 24)
    mae_naive1.append(np.mean(np.abs(daily_price[T1*24 + h::24] - pf_naive1[T1*24 + h::24])))
    rmse_naive1.append(np.sqrt(np.mean((daily_price[T1*24 + h::24] - pf_naive1[T1*24 + h::24]) ** 2)))

    # 2nd Naive Forecast
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['day_of_week'] = df['YYYYMMDD'].dt.day_name()
    pf_naive2 = df['Zonal price'].shift(1 * 24)
    included_days = ['Monday', 'Saturday', 'Sunday']
    df['Naive forecast 2'] = df['Zonal price']
    df.loc[df['day_of_week'].isin(included_days), 'Naive forecast 2'] = df['Zonal price'].shift(7 * 24)
    pf_naive2 = df['Naive forecast 2'].values
    mae_naive2.append(np.mean(np.abs(daily_price[T1*24 + h::24] - pf_naive2[T1*24 + h::24])))
    rmse_naive2.append(np.sqrt(np.mean((daily_price[T1*24 + h::24] - pf_naive2[T1*24 + h::24]) ** 2)))

    # 3rd Naive Forecast
    pf_naive3 = df['Zonal price'].shift(1 * 24)
    mae_naive3.append(np.mean(np.abs(daily_price[T1*24 + h::24] - pf_naive3[T1*24 + h::24])))
    rmse_naive3.append(np.sqrt(np.mean((daily_price[T1*24 + h::24] - pf_naive3[T1*24 + h::24]) ** 2)))

# DAILY AVERAGE
# Calculate the daily average price
daily_avg_price = df.groupby('YYYYMMDD')['Zonal price'].mean()
daily_avg_day_of_week = df.groupby('YYYYMMDD')['Day of the week'].first()
daily_avg_combined = pd.concat([daily_avg_price, daily_avg_day_of_week], axis=1)

# Initialize lists to store MAE and RMSE for Holt-Winters and Naive methods
mae_hw_daily = [0] * len(daily_avg_price)
rmse_hw_daily = [0] * len(daily_avg_price)
mae_naive1_daily = [0] * len(daily_avg_price)
rmse_naive1_daily = [0] * len(daily_avg_price)
mae_naive2_daily = [0] * len(daily_avg_price)
rmse_naive2_daily = [0] * len(daily_avg_price)
mae_naive3_daily = [0] * len(daily_avg_price)
rmse_naive3_daily = [0] * len(daily_avg_price)

mae_hw_daily_long = [0] * len(daily_avg_price)
rmse_hw_daily_long = [0] * len(daily_avg_price)

# Holt-Winters
param = fmin(holtwinters, initial_param, args=(s, daily_avg_price[0:T]))
pf_hw = holtwinters_forecast(param, s, daily_avg_price[T:])

param_long = fmin(holtwinters, initial_param, args=(s, daily_avg_price[0:T1]))
pf_hw_long = holtwinters_forecast(param, s, daily_avg_price[T1:])

mae_hw_daily.append(np.mean(np.abs(daily_avg_price[T1 + 1:] - pf_hw)))
rmse_hw_daily.append(np.sqrt(np.mean((daily_avg_price[T1 + 1:] - pf_hw) ** 2)))
mae_hw_daily_long.append(np.mean(np.abs(daily_avg_price[T1 + 1:] - pf_hw_long)))
rmse_hw_daily_long.append(np.sqrt(np.mean((daily_avg_price[T1 + 1:] - pf_hw_long) ** 2)))

# 1st Naive Forecast
pf_naive1_daily = daily_avg_price.shift(7)
mae_naive1_daily.append(np.mean(np.abs(daily_avg_price[-len(pf_hw_long):] - pf_naive1_daily[-len(pf_hw_long):])))
rmse_naive1_daily.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw_long):] - pf_naive1_daily[-len(pf_hw_long):]) ** 2)))

# 2nd Naive Forecast
pf_naive2_daily = daily_avg_combined['Zonal price'].shift(1) #domyślnie
included_days = [1.0, 6.0, 7.0]
# Update values in pf_naive2_daily based on the condition
mask = daily_avg_combined['Day of the week'].isin(included_days)
pf_naive2_daily[mask] = pf_naive1_daily[mask]
mae_naive2_daily.append(np.mean(np.abs(daily_avg_price[-len(pf_hw_long):] - pf_naive2_daily[-len(pf_hw_long):])))
rmse_naive2_daily.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw_long):] - pf_naive2_daily[-len(pf_hw_long):]) ** 2)))

# 3rd Naive Forecast
pf_naive3_daily = daily_avg_price.shift(1)
mae_naive3_daily.append(np.mean(np.abs(daily_avg_price[-len(pf_hw_long):] - pf_naive3_daily[-len(pf_hw_long):])))
rmse_naive3_daily.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw_long):] - pf_naive3_daily[-len(pf_hw_long):]) ** 2)))

'''A longer calibration window can lead to more accurate forecasts when the historical data contains substantial 
patterns, trends, and seasonality that can be captured by the forecasting model. In this case, 
having a larger dataset to estimate model parameters can result in more accurate predictions. 
The errors MAE and RMSE are smaller so forecasts are more accurate.'''

mae_hw_long = []
rmse_hw_long = []
mae_naive1_long = []
rmse_naive1_long = []
mae_naive2_long = []
rmse_naive2_long= []
mae_naive3_long= []
rmse_naive3_long = []

for h in range(1, 25):
    # Filter data for the current hour
    p_hour = df[df['HH'] == h]['Zonal price'].values
    daily_price = df['Zonal price'].values

    # Holt-Winters
    param = fmin(holtwinters, initial_param, args=(s, p_hour[0:T1]))
    pf_hw = holtwinters_forecast(param, s, p_hour[T1:])
    mae_hw_long.append(np.mean(np.abs(p_hour[T1:] - pf_hw)))
    rmse_hw_long.append(np.sqrt(np.mean((p_hour[T1:] - pf_hw) ** 2)))

    # 1st Naive Forecast
    pf_naive1 = df['Zonal price'].shift(7 * 24)
    mae_naive1_long.append(np.mean(np.abs(daily_price[T1*24 + h::24] - pf_naive1[T1*24 + h::24])))
    rmse_naive1_long.append(np.sqrt(np.mean((daily_price[T1*24 + h::24] - pf_naive1[T1*24 + h::24]) ** 2)))

    # 2nd Naive Forecast
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['day_of_week'] = df['YYYYMMDD'].dt.day_name()
    pf_naive2 = df['Zonal price'].shift(1 * 24)
    included_days = ['Monday', 'Saturday', 'Sunday']
    df['Naive forecast 2'] = df['Zonal price']
    df.loc[df['day_of_week'].isin(included_days), 'Naive forecast 2'] = df['Zonal price'].shift(7 * 24)
    pf_naive2 = df['Naive forecast 2'].values
    mae_naive2_long.append(np.mean(np.abs(daily_price[T1*24 + h::24] - pf_naive2[T1*24 + h::24])))
    rmse_naive2_long.append(np.sqrt(np.mean((daily_price[T1*24 + h::24] - pf_naive2[T1*24 + h::24]) ** 2)))

    # 3rd Naive Forecast
    pf_naive3 = df['Zonal price'].shift(1 * 24)
    mae_naive3_long.append(np.mean(np.abs(daily_price[T1*24 + h::24] - pf_naive3[T1*24 + h::24])))
    rmse_naive3_long.append(np.sqrt(np.mean((daily_price[T1*24 + h::24] - pf_naive3[T1*24 + h::24]) ** 2)))

# DAILY AVERAGE
# Calculate the daily average price
daily_avg_price = df.groupby('YYYYMMDD')['Zonal price'].mean()
daily_avg_day_of_week = df.groupby('YYYYMMDD')['Day of the week'].first()
daily_avg_combined = pd.concat([daily_avg_price, daily_avg_day_of_week], axis=1)

# Initialize lists to store MAE and RMSE for Naive methods
mae_naive1_daily_long = [0] * len(daily_avg_price)
rmse_naive1_daily_long = [0] * len(daily_avg_price)
mae_naive2_daily_long = [0] * len(daily_avg_price)
rmse_naive2_daily_long = [0] * len(daily_avg_price)
mae_naive3_daily_long = [0] * len(daily_avg_price)
rmse_naive3_daily_long = [0] * len(daily_avg_price)

# 1st Naive Forecast
pf_naive1_daily = daily_avg_price.shift(7)
mae_naive1_daily_long.append(np.mean(np.abs(daily_avg_price[-len(pf_hw_long):] - pf_naive1_daily[-len(pf_hw_long):])))
rmse_naive1_daily_long.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw_long):] - pf_naive1_daily[-len(pf_hw_long):]) ** 2)))
# prognozy z HW mają 728 dni - tutaj wybieramy oststnie 728 wartosci prognoz dla naive
# dzieki temu bedziemy porownywac modele na tych samych okresach

# 2nd Naive Forecast
pf_naive2_daily = daily_avg_combined['Zonal price'].shift(1) #domyślnie
included_days = [1.0, 6.0, 7.0]
# Update values in pf_naive2_daily based on the condition
mask = daily_avg_combined['Day of the week'].isin(included_days)
pf_naive2_daily[mask] = pf_naive1_daily[mask]
mae_naive2_daily_long.append(np.mean(np.abs(daily_avg_price[-len(pf_hw_long):] - pf_naive2_daily[-len(pf_hw_long):])))
rmse_naive2_daily_long.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw_long):] - pf_naive2_daily[-len(pf_hw_long):]) ** 2)))
#pokazuje tak jak w 3, sprawdzić, dodać rmse, sumować wszystkie listy i plotować sume

# 3rd Naive Forecast
pf_naive3_daily = daily_avg_price.shift(1)
mae_naive3_daily_long.append(np.mean(np.abs(daily_avg_price[-len(pf_hw_long):] - pf_naive3_daily[-len(pf_hw_long):])))
rmse_naive3_daily_long.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw_long):] - pf_naive3_daily[-len(pf_hw_long):]) ** 2)))

#sprawdzenie
mae_hw = np.nansum(mae_hw)
mae_hw_long = np.nansum(mae_hw_long)
mae_hw_daily = np.nansum(mae_hw_daily)
mae_hw_daily_long = np.nansum(mae_hw_daily_long)

mae_naive1 = sum(mae_naive1)
mae_naive1_long = sum(mae_naive1_long)
mae_naive1_daily = sum(mae_naive1_daily)
mae_naive1_daily_long = sum(mae_naive1_daily_long)

mae_naive2 = sum(mae_naive2)
mae_naive2_long = sum(mae_naive2_long)
mae_naive2_daily = sum(mae_naive2_daily)
mae_naive2_daily_long = sum(mae_naive2_daily_long)

mae_naive3 = sum(mae_naive3)
mae_naive3_long = sum(mae_naive3_long)
mae_naive3_daily = sum(mae_naive3_daily)
mae_naive3_daily_long = sum(mae_naive3_daily_long)

print(mae_hw, mae_hw_long) #jest inne, duża różnica
print(mae_naive1, mae_naive1_long) #te same wartości
print(mae_naive2, mae_naive2_long) #same
print(mae_naive3, mae_naive3_long) #same
print(mae_hw_daily, mae_hw_daily_long) #jest inne, mała różnica
print(mae_naive1_daily, mae_naive1_daily_long) #same
print(mae_naive2_daily, mae_naive2_daily_long) #same
print(mae_naive3_daily, mae_naive3_daily_long) #same

#jest sens porównywać Naive, dla których się nic nie zmienia?
#wyniki