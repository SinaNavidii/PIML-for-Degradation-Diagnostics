import GPy
import numpy as np
import pandas as pd
from GPy.core.parameterization import Param

class CoKrigingModel:
    def __init__(self, data_path, train_indices, test_indices, feature_cols, target_cols):
        """
        Initialize the Co-Kriging Model with given data and indices.
        """
        # Load and split the data
        self.data_path = data_path
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.X_train, self.y_train, self.X_test, self.y_test = self.get_data_and_split()

        # Define kernels for low-fidelity and high-fidelity models
        self.kernel_low = GPy.kern.Matern32(input_dim=self.X_train.shape[1])
        self.kernel_high = GPy.kern.Matern32(input_dim=self.X_train.shape[1])

        # Define trainable rho parameter for cross-covariance
        self.rho = Param('rho', 1.0)  # Initial value 1.0
        self.kernel_cross = self.rho * self.kernel_low  # Cross covariance

        # Construct the Co-Kriging model with joint covariance structure
        self.model = GPy.models.GPCoregionalizedRegression(
            [self.X_train, self.X_train],  # Low and High fidelity inputs
            [self.y_train, self.y_train],  # Corresponding outputs
            kernel=[self.kernel_low, self.kernel_high, self.kernel_cross]
        )

        # Attach rho as a trainable parameter
        self.model.link_parameter(self.rho)

    def get_data_and_split(self):
        """
        Load data from CSV and split it into training and testing sets.
        """
        data = pd.read_csv(self.data_path)
        df_X_train = data.iloc[self.train_indices, self.feature_cols]
        df_y_train = data.iloc[self.train_indices, self.target_cols]
        X_train = df_X_train.to_numpy()
        y_train = df_y_train.to_numpy()

        df_X_test = data.iloc[self.test_indices, self.feature_cols]
        df_y_test = data.iloc[self.test_indices, self.target_cols]
        X_test = df_X_test.to_numpy()
        y_test = df_y_test.to_numpy()

        return X_train, y_train, X_test, y_test

    def train_model(self):
        """
        Train the Co-Kriging model, optimizing rho and kernel hyperparameters.
        """
        # Optimize the model (including rho)
        self.model.optimize()
        print(f"Optimized rho: {self.rho.values}")

    def predict(self, X_test):
        """
        Make predictions with the trained Co-Kriging model.
        """
        y_pred, y_var = self.model.predict(X_test)
        return y_pred, np.sqrt(y_var)  # Return mean and standard deviation

    def run(self):
        """
        Train the Co-Kriging model and evaluate it on the test set.
        """
        self.train_model()

        # Make predictions on the test set
        y_pred, y_std = self.predict(self.X_test)

        # Calculate RMSPE (Root Mean Square Percentage Error)
        rmspe = np.sqrt(np.mean(np.square((self.y_test - y_pred) / self.y_test), axis=0))

        return rmspe, y_pred, y_std

def test_cokriging(data_path, train_indices, test_indices, feature_cols, target_cols):
    """
    Function to test the Co-Kriging model.
    """
    cokriging_model = CoKrigingModel(data_path, train_indices, test_indices, feature_cols, target_cols)
    rmspe, y_pred, y_std = cokriging_model.run()

    print(f"RMSPE: {rmspe}")
    return y_pred, y_std  # Return predictions and uncertainties

