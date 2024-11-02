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
#T1 = 720 #longer calibration window

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

    mae = np.mean(np.abs(data[:] - y[:]))
    return mae, y

# Function to calculate MAE and RMSE for a specific model
def calculate_mae_rmse(data, forecast):
    mae = np.mean(np.abs(data - forecast))
    rmse = np.sqrt(np.mean((data - forecast) ** 2))
    return mae, rmse

for h in range(1, 25):
    # Filter data for the current hour
    p_hour = df[df['HH'] == h]['Zonal price'].values
    daily_price = df['Zonal price'].values
    #print(daily_price.size)

    # Holt-Winters
    param = fmin(holtwinters, initial_param, args=(s, p_hour[0:T]))
    MAE, pf_hw = holtwinters_forecast(param, s, p_hour[T:])
    mae_hw.append(MAE) # czy tu jest teraz okej to MAE z tej funkcji? Czy w funkcji jest okej??
    rmse_hw.append(np.sqrt(np.mean((p_hour[T:] - pf_hw) ** 2)))
    #print(mae_hw, rmse_hw)

    # 1st Naive Forecast
    pf_naive1 = df['Zonal price'].shift(7 * 24)
    #print(pf_naive1.size)
    mae_naive1.append(np.mean(np.abs(daily_price[T*24 + h::24] - pf_naive1[T*24 + h::24])))
    rmse_naive1.append(np.sqrt(np.mean((daily_price[T*24 + h::24] - pf_naive1[T*24 + h::24]) ** 2)))
    #print(mae_naive1, rmse_naive1)

    # 2nd Naive Forecast
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['day_of_week'] = df['YYYYMMDD'].dt.day_name()
    #print(df['day_of_week'])
    pf_naive2 = df['Zonal price'].shift(1 * 24)
    included_days = ['Monday', 'Saturday', 'Sunday']
    df['Naive forecast 2'] = df['Zonal price']
    df.loc[df['day_of_week'].isin(included_days), 'Naive forecast 2'] = df['Zonal price'].shift(7 * 24)
    pf_naive2 = df['Naive forecast 2'].values
    mae_naive2.append(np.mean(np.abs(daily_price[T*24 + h::24] - pf_naive2[T*24 + h::24])))
    rmse_naive2.append(np.sqrt(np.mean((daily_price[T*24 + h::24] - pf_naive2[T*24 + h::24]) ** 2)))
    #print(mae_naive2, rmse_naive2)

    # 3rd Naive Forecast
    pf_naive3 = df['Zonal price'].shift(1 * 24)
    mae_naive3.append(np.mean(np.abs(daily_price[T*24 + h::24] - pf_naive3[T*24 + h::24])))
    rmse_naive3.append(np.sqrt(np.mean((daily_price[T*24 + h::24] - pf_naive3[T*24 + h::24]) ** 2)))
    #print(mae_naive3, rmse_naive3)

# DAILY AVERAGE
# Calculate the daily average price
daily_avg_price = df.groupby('YYYYMMDD')['Zonal price'].mean()
daily_avg_day_of_week = df.groupby('YYYYMMDD')['Day of the week'].first()
daily_avg_combined = pd.concat([daily_avg_price, daily_avg_day_of_week], axis=1)
print(daily_avg_combined)

# Initialize lists to store MAE and RMSE for Holt-Winters and Naive methods
mae_hw_daily = [0] * len(daily_avg_price)
rmse_hw_daily = [0] * len(daily_avg_price)
mae_naive1_daily = [0] * len(daily_avg_price)
rmse_naive1_daily = [0] * len(daily_avg_price)
mae_naive2_daily = [0] * len(daily_avg_price)
rmse_naive2_daily = [0] * len(daily_avg_price)
mae_naive3_daily = [0] * len(daily_avg_price)
rmse_naive3_daily = [0] * len(daily_avg_price)

T=360
# Holt-Winters
param = fmin(holtwinters, initial_param, args=(s, daily_avg_price[0:T]))
MAE, pf_hw = holtwinters_forecast(param, s, daily_avg_price[T:])
mae_hw_daily.append(MAE)
rmse_hw_daily.append(np.sqrt(np.mean((daily_avg_price[T + 1:] - pf_hw) ** 2)))
# pf_hw powinno miec tutaj odpowiednia dlugosc - jestrobione na podstawie daily_avg_price[T:]

# 1st Naive Forecast
pf_naive1_daily = daily_avg_price.shift(7)
mae_naive1_daily.append(np.mean(np.abs(daily_avg_price[-len(pf_hw):] - pf_naive1_daily[-len(pf_hw):])))
rmse_naive1_daily.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw):] - pf_naive1_daily[-len(pf_hw):]) ** 2)))
# prognozy z HW mają 728 dni - tutaj wybieramy oststnie 728 wartosci prognoz dla naive
# dzieki temu bedziemy porownywac modele na tych samych okresach


# 2nd Naive Forecast
pf_naive2_daily = daily_avg_combined['Zonal price'].shift(1) #domyślnie
print(pf_naive2_daily)
included_days = [1.0, 6.0, 7.0]
# Update values in pf_naive2_daily based on the condition
mask = daily_avg_combined['Day of the week'].isin(included_days)
pf_naive2_daily[mask] = pf_naive1_daily[mask]
mae_naive2_daily.append(np.mean(np.abs(daily_avg_price[-len(pf_hw):] - pf_naive2_daily[-len(pf_hw):])))
rmse_naive2_daily.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw):] - pf_naive2_daily[-len(pf_hw):]) ** 2)))
#pokazuje tak jak w 3, sprawdzić, dodać rmse, sumować wszystkie listy i plotować sume

# 3rd Naive Forecast
pf_naive3_daily = daily_avg_price.shift(1)
mae_naive3_daily.append(np.mean(np.abs(daily_avg_price[-len(pf_hw):] - pf_naive3_daily[-len(pf_hw):])))
rmse_naive3_daily.append(np.sqrt(np.mean((daily_avg_price[-len(pf_hw):] - pf_naive3_daily[-len(pf_hw):]) ** 2)))

# Create plots for MAE and RMSE for each model
plt.figure(figsize=(10, 5))
plt.plot(range(1, 25), mae_hw, label='HW', marker='o')
plt.plot(range(1, 25), mae_naive1, label='Naive 1', marker='o')
plt.plot(range(1, 25), mae_naive2, label='Naive 2', marker='o')
plt.plot(range(1, 25), mae_naive3, label='Naive 3', marker='o')
plt.xlabel('Hour (h)')
plt.ylabel('MAE')
plt.title('Average MAE for all forecasts')
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(range(1, 25), rmse_hw, label='HW', marker='o')
plt.plot(range(1, 25), rmse_naive1, label='Naive 1', marker='o')
plt.plot(range(1, 25), rmse_naive2, label='Naive 2', marker='o')
plt.plot(range(1, 25), rmse_naive3, label='Naive 3', marker='o')
plt.xlabel('Hour (h)')
plt.ylabel('RMSE')
plt.title('Average RSME for all forecasts')
plt.legend()
plt.show()

#sprawdzenie
mae_hw_daily = sum(mae_hw_daily)
rmse_hw_daily = sum(rmse_hw_daily)
mae_naive1_daily = sum(mae_naive1_daily)
rmse_naive1_daily = sum(rmse_naive1_daily)
mae_naive2_daily = sum(mae_naive2_daily)
rmse_naive2_daily = sum(rmse_naive2_daily)
mae_naive3_daily = sum(mae_naive3_daily)
rmse_naive3_daily = sum(rmse_naive3_daily)

print("MAE: ")
print(mae_hw_daily)
print(mae_naive1_daily)
print(mae_naive2_daily)
print(mae_naive3_daily)

print("RMSE: ")
print(rmse_hw_daily)
print(rmse_naive1_daily)
print(rmse_naive2_daily)
print(rmse_naive3_daily)

# Create bar charts for MAE
mae_values = [mae_hw_daily, mae_naive1_daily, mae_naive2_daily, mae_naive3_daily]
forecast_methods = ['Holt-Winters', 'Naive 1', 'Naive 2', 'Naive 3']
plt.figure(figsize=(10, 5))
plt.bar(forecast_methods, mae_values, color='skyblue')
plt.ylabel('MAE')
plt.title('Average MAE for all forecasts, daily')

# Create bar charts for RMSE
rmse_values = [rmse_hw_daily, rmse_naive1_daily, rmse_naive2_daily, rmse_naive3_daily]
plt.figure(figsize=(10, 5))
plt.bar(forecast_methods, rmse_values, color='lightcoral')
plt.ylabel('RMSE')
plt.title('Average RMSE for all forecasts, daily')
plt.show()