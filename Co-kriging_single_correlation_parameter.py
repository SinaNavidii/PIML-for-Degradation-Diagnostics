import pandas as pd
import numpy as np
import GPy
from scipy.optimize import minimize

class CokrigingModel:
    def __init__(self, data_path, train_indices, test_indices, feature_cols, target_cols):
        self.data_path = data_path
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.X_train, self.y_train, self.X_test, self.y_test = self.get_data_and_split()

        # Create Gaussian Process Regression models with Matern kernels
        self.kernel_low = GPy.kern.Matern32(input_dim=self.X_train.shape[1])
        self.kernel_high = GPy.kern.Matern32(input_dim=self.X_train.shape[1])

        self.gaussian_process_1 = GPy.models.GPRegression(self.X_train, self.y_train, self.kernel_low)
        self.gaussian_process_d = GPy.models.GPRegression(self.X_train, self.y_train, self.kernel_high)

    def get_data_and_split(self):
        """Loads and splits the data into training and testing sets."""
        data = pd.read_csv(self.data_path) # Assuming that data is in csv format!

        df_X_train = data.iloc[self.train_indices, self.feature_cols]
        df_y_train = data.iloc[self.train_indices, self.target_cols]
        X_train = df_X_train.to_numpy()
        y_train = df_y_train.to_numpy()

        df_X_test = data.iloc[self.test_indices, self.feature_cols]
        df_y_test = data.iloc[self.test_indices, self.target_cols]
        X_test = df_X_test.to_numpy()
        y_test = df_y_test.to_numpy()

        return X_train, y_train, X_test, y_test

    def optimize_rho(self):
        """Optimizes rho considering both GP models simultaneously."""
        def objective(rho):
            # Adjusted predictions based on rho
            y_d = self.gaussian_process_1.predict(self.X_train)[0] - rho * self.y_train
            self.gaussian_process_d.set_Y(y_d)

            # Combine the log-likelihoods of both GP models
            log_likelihood_1 = self.gaussian_process_1.log_likelihood()
            log_likelihood_d = self.gaussian_process_d.log_likelihood()

            # Return the negative sum of log-likelihoods (to minimize)
            return -(log_likelihood_1 + log_likelihood_d)

        initial_rho = 0.0
        result = minimize(objective, x0=initial_rho, bounds=[(-1, 1)])  # Adjust bounds as necessary
        return result.x[0]

    def run(self):
        """Main method to execute the Cokriging process with simultaneous GP model training."""
        # Train both models simultaneously by optimizing rho
        rho = self.optimize_rho()

        # Make predictions on the test data using the first model
        y_pred_e, _ = self.gaussian_process_1.predict(self.X_test)

        # Calculate derived values based on optimized rho
        y_d = y_pred_e - rho * self.y_train  # Adjusted predictions

        # Set the second model's predictions
        self.gaussian_process_d.set_Y(y_d)

        # Train both models (with updated data) after optimization of rho
        self.gaussian_process_1.optimize()
        self.gaussian_process_d.optimize()

        # Make final predictions from both models
        y_c_final, _ = self.gaussian_process_1.predict(self.X_test)
        y_d_final, _ = self.gaussian_process_d.predict(self.X_test)

        # Calculate the final result
        y_e_final = rho * y_c_final + y_d_final

        # Calculate RMSPE
        rmspe = np.sqrt(np.mean(np.square((self.y_test - y_e_final) / self.y_test), axis=0))
        return rmspe

def test_cokriging(data_path, train_indices, test_indices, feature_cols, target_cols):
    """Function to run and test the Cokriging model."""
    cokriging_model = CokrigingModel(data_path, train_indices, test_indices, feature_cols, target_cols)
    rmspe = cokriging_model.run()
    print(f"RMSPE: {rmspe}")
