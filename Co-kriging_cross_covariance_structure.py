import GPy
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class CoKrigingModel:
    def __init__(self, data_path, train_indices, test_indices, feature_cols, target_cols):
        # Initialization of parameters
        self.data_path = data_path
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.X_train, self.y_train, self.X_test, self.y_test = self.get_data_and_split()

        # Define kernel for both GP processes with a cross-covariance term
        self.kernel_low = GPy.kern.Matern32(input_dim=self.X_train.shape[1])
        self.kernel_high = GPy.kern.Matern32(input_dim=self.X_train.shape[1])
        
        # Define a joint covariance matrix for co-kriging (auto and cross covariance)
        self.kernel_cross = GPy.kern.Matern32(input_dim=self.X_train.shape[1])  # Cross-covariance kernel
        
        self.model = GPy.models.GPCoregionalizedRegression(
            [self.X_train, self.X_train],
            [self.y_train, self.y_train],
            kern=[self.kernel_low, self.kernel_high, self.kernel_cross]
        )

    def get_data_and_split(self):
        """Loads and splits the data into training and testing sets."""
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
        """Train the Co-Kriging model."""
        # Optimize the model (this includes both auto and cross-covariances)
        self.model.optimize()

    def predict(self, X_test):
        """Make predictions with the trained model."""
        return self.model.predict(X_test)

    def run(self):
        """Run the Co-Kriging model for prediction and error calculation."""
        self.train_model()

        # Make predictions on the test set
        y_pred, _ = self.predict(self.X_test)

        # Calculate RMSPE (Root Mean Square Percentage Error)
        rmspe = np.sqrt(np.mean(np.square((self.y_test - y_pred[0]) / self.y_test), axis=0))
        return rmspe

def test_cokriging(data_path, train_indices, test_indices, feature_cols, target_cols):
    """Function to test the Co-Kriging model."""
    cokriging_model = CoKrigingModel(data_path, train_indices, test_indices, feature_cols, target_cols)
    rmspe = cokriging_model.run()
    print(f"RMSPE: {rmspe}")
