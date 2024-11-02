import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import fmin

d = np.loadtxt('GEFCOM.txt')
dataset = d[:, 2]

def calculate_mae_rmse(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) between true and predicted values.
    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    Returns:
    - MAE: Mean Absolute Error.
    - RMSE: Root Mean Squared Error.
    """
    absolute_errors = np.abs(y_true - y_pred)
    squared_errors = (y_true - y_pred) ** 2
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    return mae, rmse


def forecastETS(param, x):
    alpha = param[0]

    # Initialize L and FX vectors
    L = np.zeros(len(x))
    fx = np.zeros(len(x))

    # Set initial values of L and FX
    L[0] = x[0]
    fx[1] = L[0]

    # Iterate to compute L(t) and FX(t)
    for t in range(1, len(x) - 1):
        L[t] = alpha * x[t] + (1 - alpha) * L[t - 1]
        fx[t + 1] = L[t]

    return fx

def simpleETS(param, x):
    alpha = param[0]

    # Initialize L and FX vectors
    L = np.zeros(len(x))
    fx = np.zeros(len(x))

    # Set initial values of L and FX
    L[0] = x[0]
    fx[1] = L[0]

    # Iterate to compute L(t) and FX(t)
    for t in range(1, len(x) - 1):
        L[t] = alpha * x[t] + (1 - alpha) * L[t - 1]
        fx[t + 1] = L[t]

    # Compute MSE + a penalty for parameters beyond the admitted range
    maxx = np.max(x)
    mse = np.mean((x[1:] - fx[1:]) ** 2) + maxx * (alpha <= 0) + maxx * (alpha >= 1)
    #print(fx)

    return mse

initial_param = [0.5]
T = 360
mae_ets = []
rmse_ets = []
mae_naive1 = []
mae_naive2 = []
mae_naive3 = []
rmse_naive1 = []
rmse_naive2 = []
rmse_naive3 = []

day_of_week = d[:, 5]


for h in range(24):
    data_h = dataset[h::24]
    train_data = data_h[:T]
    test_data = data_h[T:]
    result = minimize(simpleETS, initial_param, args=(data_h[:T]), bounds=[(0, 1)])
    opt_alpha = result.x
    forecast = forecastETS(opt_alpha, data_h[T:])
    #print(forecast.size)
    #print(test_data.size)
    mae_ets.append(np.mean(np.abs(forecast-test_data)))
    rmse_ets.append(np.sqrt(np.mean((forecast-test_data) ** 2)))

    pf_naive1 = np.roll(test_data, -7)
    mae_naive1.append(np.mean(np.abs(test_data - pf_naive1)))
    rmse_naive1.append(np.sqrt(np.mean((test_data - pf_naive1) ** 2)))

    day_of_week_h = d[h::24]
    day_of_week = day_of_week_h[T:, 5]
    pf_naive2 = test_data
    shift_amount = np.where((day_of_week == 1) | (day_of_week == 6) | (day_of_week == 7), -7, -1)
    # Roll 'pf_naive2' based on the calculated shift amount
    pf_naive2 = np.roll(pf_naive2, shift_amount)
    mae_naive2.append(np.mean(np.abs(test_data - pf_naive2)))
    rmse_naive2.append(np.sqrt(np.mean((test_data - pf_naive2) ** 2)))

    pf_naive3 = np.roll(test_data, -1)
    mae_naive3.append(np.mean(np.abs(test_data - pf_naive3)))
    rmse_naive3.append(np.sqrt(np.mean((test_data - pf_naive3) ** 2)))

plt.figure(1)
plt.plot(mae_ets, '.-', label='ETS')
plt.plot(mae_naive1, '.-', label='Naive1')
plt.plot(mae_naive2, '.-', label='Naive2')
plt.plot(mae_naive3, '.-', label='Naive3')
plt.xlabel('Hour (h)')
plt.ylabel('MAE')
plt.title('Average MAE for all forecasts')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(rmse_ets, '.-', label='ETS')
plt.plot(rmse_naive1, '.-', label='Naive1')
plt.plot(rmse_naive2, '.-', label='Naive2')
plt.plot(rmse_naive3, '.-', label='Naive3')
plt.xlabel('Hour (h)')
plt.ylabel('RMSE')
plt.title('Average RMSE for all forecasts')
plt.legend()
plt.show()

# Average daily
df2 = pd.DataFrame(d, columns=['YYYYMMDD', 'HH', 'Zonal price', 'System load', 'Zonal load', 'Day of the week'])
daily_avg_price = df2.groupby('YYYYMMDD')['Zonal price'].mean()
data_train2 = daily_avg_price[:T]
data_test2 = daily_avg_price[T:]
result2 = minimize(simpleETS, initial_param, args=(daily_avg_price.values[:T]), bounds=[(0, 1)])
opt_alpha2 = result2.x
forecast2 = forecastETS(opt_alpha2, daily_avg_price.values[T:])
# Calculate errors
# Find the common length for data_test2 and forecast2
data_test2_list = data_test2.tolist()
common_length = min(len(data_test2_list), len(forecast2))
mae_ets, rmse_ets = calculate_mae_rmse(data_test2_list[:common_length], forecast2[:common_length])
print("Daily: ")
print('MAE:', mae_ets)
print('RMSE:', rmse_ets)