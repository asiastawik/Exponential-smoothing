import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load Algeria Exports data
data = np.loadtxt('Algeria_Exports.txt', skiprows=1)
p = data[:, 1]

# Last day of the calibration period
T = 58

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

def forecastingETS(param, x):
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

# Estimate Simple ETS parameters
initial_param = [0.5]
result = minimize(simpleETS, initial_param, args=(p[:T]), bounds=[(0, 1)]) #aby było pomiędzy 0 a 1
param = result.x
print('   alpha')
print(param)

# Compute Simple ETS forecasts
pf = forecastingETS(param, p[1:])
print(pf)

print('   MAE')
print(np.mean(np.abs(p[2:] - pf[1:])))
print('   MSE')
print(np.mean((p[2:] - pf[1:]) ** 2))

plt.figure(1)
plt.plot(p[1:], '.-', label='Real data')
plt.plot(pf[1:], '.-', label='Forecasts')
plt.legend()
plt.show()