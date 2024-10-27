import pandas as pd
import numpy as np
import GPy
from scipy.optimize import minimize

# Function to get data and split into training and testing sets
def get_data_and_split(data_path, train_indices, test_indices, feature_cols, target_cols):
    # Load data
    data = pd.read_csv(data_path)

    # Prepare training data
    df_X_train = data.iloc[train_indices, feature_cols]
    df_y_train = data.iloc[train_indices, target_cols]
    X_train = df_X_train.to_numpy()
    y_train = df_y_train.to_numpy()

    # Prepare testing data
    df_X_test = data.iloc[test_indices, feature_cols]
    df_y_test = data.iloc[test_indices, target_cols]
    X_test = df_X_test.to_numpy()
    y_test = df_y_test.to_numpy()

    return X_train, y_train, X_test, y_test

# Function to optimize rho based on log-likelihood
def optimize_rho(gaussian_process_1, gaussian_process_d, X_train, y_train):
    def objective(rho):
        y_d = gaussian_process_1.predict(X_train)[0] - rho * y_train
        gaussian_process_d.set_Y(y_d)
        return -gaussian_process_d.log_likelihood()  # Negative log-likelihood for minimization

    # Initial guess for rho
    initial_rho = 0.0
    result = minimize(objective, x0=initial_rho, bounds=[(-1, 1)])  # Adjust bounds as necessary
    return result.x[0]

# Main function to execute the GP regression process
def run_gp_regression(data_path, train_indices, test_indices, feature_cols, target_cols):
    # Get training and testing sets
    X_train, y_train, X_test, y_test = get_data_and_split(data_path, train_indices, test_indices, feature_cols, target_cols)

    # Create Gaussian Process Regression model with Matern kernel for training data
    kernel_low = GPy.kern.Matern32(input_dim=X_train.shape[1])
    gaussian_process_1 = GPy.models.GPRegression(X_train, y_train, kernel_low)

    # Train the model
    gaussian_process_1.optimize()

    # Create a second Gaussian Process model for the derived output
    kernel_high = GPy.kern.Matern32(input_dim=X_train.shape[1])
    gaussian_process_d = GPy.models.GPRegression(X_train, y_train, kernel_high)

    # Optimize rho based on log-likelihood
    rho = optimize_rho(gaussian_process_1, gaussian_process_d, X_train, y_train)

    # Make predictions on the test data using the first model
    y_pred_e, _ = gaussian_process_1.predict(X_test)

    # Calculate derived values based on optimized rho
    y_d = y_pred_e - rho * y_train  # Adjusted predictions

    # Set the second model's predictions
    gaussian_process_d.set_Y(y_d)

    # Train the second model
    gaussian_process_d.optimize()

    # Make final predictions
    y_c_final, _ = gaussian_process_1.predict(X_test)
    y_d_final, _ = gaussian_process_d.predict(X_test)

    # Calculate the final result
    y_e_final = rho * y_c_final + y_d_final

    # Calculate RMSPE
    rmspe = np.sqrt(np.mean(np.square((y_test - y_e_final) / y_test), axis=0))
    return rmspe