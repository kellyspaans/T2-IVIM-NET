from scipy.optimize import least_squares
import numpy as np

# Function to rescale parameters from [0, 1] to their original range
def rescale_params(scaled_params, bounds):
    return [p * (upper - lower) + lower for p, (lower, upper) in zip(scaled_params, bounds)]

# T2-IVIM model function with rescaled parameters
def T2_ivim(bvalues, TEs, scaled_params, bounds):
    Fp0, Dt, Fp, Dp, T2p, T2t = rescale_params(scaled_params, bounds)
    return (Fp * np.exp(-bvalues * Dp) * np.exp(-TEs / T2p) + (Fp0) * np.exp(-bvalues * Dt) * np.exp(-TEs / T2t))

# Residuals function
def residuals(scaled_params, bvalues, TEs, signal, bounds):
    return signal - T2_ivim(bvalues, TEs, scaled_params, bounds)

# Bounds for each parameter
original_bounds = [
    (0, 1),     # Fp
    (0, 0.003), # D
    (0, 0.05),  # D*
    (0.04, 0.12), # T2t
    (0.02, 0.15), # T2p
    (0, 2.5)    # Fp0
]

# Rescale bounds to [0, 1]
scaled_bounds = [(0, 1) for _ in original_bounds]

# Example data
bvalues = np.array([...]) # Array of b-values
TEs = np.array([...])     # Array of TEs
observed_signal = np.array([...]) # Observed signal values

initial_guess = [0.5, 0.01, 0.01, 0.1, 0.1, 1]

# Rescale initial guess to [0, 1]
initial_guess_scaled = [(param - lower) / (upper - lower) for param, (lower, upper) in zip(initial_guess, original_bounds)]

# Least squares fitting with rescaled parameters
result = least_squares(residuals, initial_guess_scaled, args=(bvalues, TEs, observed_signal, original_bounds), bounds=(0, 1))

# Optimal rescaled parameters
optimal_scaled_params = result.x

# Convert back to original scale
optimal_params = rescale_params(optimal_scaled_params, original_bounds)
print("Optimal Parameters:", optimal_params)
