import numpy as np
import matplotlib.pyplot as plt

# Load Algeria Exports (2 columns: Year, Exports)
d = np.loadtxt('Algeria_Exports.txt', skiprows=1)
p = d[:, 1]

# Last day of the calibration period
T = 58

# Define a list of alpha values to iterate over
alpha_values = np.arange(0.01, 1.0, 0.01)

# Initialize variables to track the best alpha and minimum MSE
best_alpha = None
min_mse = float('inf')

# Iterate over the alpha values
for alpha in alpha_values:
    # Initialize L and FX vectors
    L = np.zeros_like(p)
    fx = np.zeros_like(p)

    # Set initial values of L and FX
    L[0] = p[0]
    fx[1] = L[0]

    # Iterate to compute L(t) and FX(t)
    for t in range(1, len(p) - 1):
        L[t] = alpha * p[t] + (1 - alpha) * L[t - 1]
        fx[t + 1] = L[t]

    # Calculate MSE
    mse = np.mean((p[1:] - fx[1:]) ** 2)

    # Update the best alpha if a lower MSE is found
    if mse < min_mse:
        best_alpha = alpha
        min_mse = mse

# Print the best alpha and corresponding MSE
print("Best alpha:", best_alpha)
print("Minimum MSE:", min_mse)

# Compute Simple ETS forecasts using the best alpha
L = np.zeros_like(p)
fx = np.zeros_like(p)
L[0] = p[0]
fx[1] = L[0]
for t in range(1, len(p) - 1):
    L[t] = best_alpha * p[t] + (1 - best_alpha) * L[t - 1]
    fx[t + 1] = L[t]

# Calculate MAE and MSE using the best alpha
mae = np.mean(np.abs(p[1:] - fx[1:]))

print('MAE:', mae)

# Plot data and forecasts
plt.figure(1)
plt.plot(p[1:], '.-', label='real')
plt.plot(fx[1:], '.-', label='forecast')
plt.show()