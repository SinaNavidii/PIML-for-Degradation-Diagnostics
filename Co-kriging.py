import GPy
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


class CoKrigingModel:
    def __init__(self, data_path, train_lf_indices, train_hf_indices, test_indices, feature_cols, target_cols):
        self.data_path = data_path
        self.train_lf_indices = train_lf_indices
        self.train_hf_indices = train_hf_indices
        self.test_indices = test_indices
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        self.X_L, self.Y_L, self.X_H, self.Y_H, self.X_test, self.Y_test = self._load_data()
        self.n_outputs = self.Y_L.shape[1]

        self.rho = np.ones(self.n_outputs)
        self.gp_L_list = [None] * self.n_outputs
        self.gp_delta_list = [None] * self.n_outputs

    def _load_data(self):
        data = pd.read_csv(self.data_path)

        def extract(indices):
            X = data.iloc[indices, self.feature_cols].to_numpy()
            Y = data.iloc[indices, self.target_cols].to_numpy()
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            return X, Y

        X_L, Y_L = extract(self.train_lf_indices)
        X_H, Y_H = extract(self.train_hf_indices)
        X_test, Y_test = extract(self.test_indices)

        return X_L, Y_L, X_H, Y_H, X_test, Y_test

    def _fit_output(self, d, rho):
        """Fit LF and delta GPs for output dimension d (Algorithm 1, Steps 1-4)."""
        y_L = self.Y_L[:, d:d+1]
        y_H = self.Y_H[:, d:d+1]

        # Step 1: train LF GP on {X_L, Y_L}
        gp_L = GPy.models.GPRegression(
            self.X_L, y_L,
            kernel=GPy.kern.Matern32(input_dim=self.X_L.shape[1])
        )
        gp_L.optimize()

        # Steps 2-3: discrepancy Y_delta = Y_H - rho * mu_L(X_H)  (Eq. 6, corrected from Algorithm 1 Step 3)
        mu_L_at_XH, _ = gp_L.predict(self.X_H)
        Y_delta = y_H - rho * mu_L_at_XH

        # Step 4: train delta GP on {X_H, Y_delta}
        gp_delta = GPy.models.GPRegression(
            self.X_H, Y_delta,
            kernel=GPy.kern.Matern32(input_dim=self.X_H.shape[1])
        )
        gp_delta.optimize()

        return gp_L, gp_delta

    def _build_covariance(self, gp_L, gp_delta, rho):
        """Build the full block covariance matrix C (Eq. 11)."""
        n_L, n_H = self.X_L.shape[0], self.X_H.shape[0]
        noise_L = float(gp_L.likelihood.variance)
        noise_delta = float(gp_delta.likelihood.variance)

        C_LL = gp_L.kern.K(self.X_L, self.X_L) + noise_L * np.eye(n_L)
        C_LH = rho * gp_L.kern.K(self.X_L, self.X_H)
        C_HH = (rho**2 * gp_L.kern.K(self.X_H, self.X_H)
                + gp_delta.kern.K(self.X_H, self.X_H)
                + noise_delta * np.eye(n_H))

        return np.block([[C_LL, C_LH], [C_LH.T, C_HH]])

    def _log_marginal_likelihood(self, rho, d):
        """Eq. (12): profile log-marginal likelihood for output d given rho."""
        gp_L, gp_delta = self._fit_output(d, rho)

        n = self.X_L.shape[0] + self.X_H.shape[0]
        C = self._build_covariance(gp_L, gp_delta, rho)
        Y = np.vstack([self.Y_L[:, d:d+1], self.Y_H[:, d:d+1]])

        try:
            L = np.linalg.cholesky(C)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            return float(-0.5 * Y.T @ alpha
                         - np.sum(np.log(np.diag(L)))
                         - n / 2 * np.log(2 * np.pi))
        except np.linalg.LinAlgError:
            return -np.inf

    def train_model(self):
        """Train all outputs: optimize rho via Eq. (12), then fit both GPs."""
        for d in range(self.n_outputs):
            result = minimize_scalar(
                lambda r, d=d: -self._log_marginal_likelihood(r, d),
                bounds=(1e-3, 10.0),
                method='bounded'
            )
            self.rho[d] = result.x
            self.gp_L_list[d], self.gp_delta_list[d] = self._fit_output(d, self.rho[d])
            print(f"Output {d}: optimized rho = {self.rho[d]:.4f}")

    def predict(self, X_test):
        """Predict using Eq. (8) for mean and Eq. (9) for variance."""
        n_test = X_test.shape[0]
        mu_all = np.zeros((n_test, self.n_outputs))
        std_all = np.zeros((n_test, self.n_outputs))

        for d in range(self.n_outputs):
            rho = self.rho[d]
            gp_L = self.gp_L_list[d]
            gp_delta = self.gp_delta_list[d]

            # Eq. (8): mean = rho * mu_L(x*) + mu_delta_posterior(x*)
            # gp_delta.predict already incorporates c_delta * C_delta^{-1} * (Y_delta - mu_delta)
            mu_L, var_L = gp_L.predict(X_test)
            mu_delta, var_delta = gp_delta.predict(X_test)
            mu_pred = rho * mu_L + mu_delta

            # Eq. (9): var = rho^2 * sigma_L^2 + sigma_delta^2 - c(x*)^T * C^{-1} * c(x*)
            # Build c(x*) per Eq. (10)
            k_L_star_XL = gp_L.kern.K(X_test, self.X_L)       # (n_test, n_L)
            k_L_star_XH = gp_L.kern.K(X_test, self.X_H)       # (n_test, n_H)
            k_delta_star_XH = gp_delta.kern.K(X_test, self.X_H)  # (n_test, n_H)

            c = np.hstack([
                rho * k_L_star_XL,
                rho**2 * k_L_star_XH + k_delta_star_XH
            ])  # (n_test, n_L + n_H)

            # Variance correction term: c^T * C^{-1} * c via Cholesky
            C = self._build_covariance(gp_L, gp_delta, rho)
            try:
                L = np.linalg.cholesky(C)
                v = np.linalg.solve(L, c.T)          # (n_L+n_H, n_test)
                correction = np.sum(v**2, axis=0).reshape(-1, 1)
            except np.linalg.LinAlgError:
                correction = np.zeros((n_test, 1))

            var_pred = np.maximum(rho**2 * var_L + var_delta - correction, 0)

            mu_all[:, d] = mu_pred.ravel()
            std_all[:, d] = np.sqrt(var_pred).ravel()

        return mu_all, std_all

    def run(self):
        self.train_model()
        y_pred, y_std = self.predict(self.X_test)
        rmspe = np.sqrt(np.mean(np.square((self.Y_test - y_pred) / self.Y_test), axis=0))
        return rmspe, y_pred, y_std


def test_cokriging(data_path, train_lf_indices, train_hf_indices, test_indices, feature_cols, target_cols):
    model = CoKrigingModel(data_path, train_lf_indices, train_hf_indices, test_indices, feature_cols, target_cols)
    rmspe, y_pred, y_std = model.run()
    print(f"RMSPE: {rmspe}")
    return y_pred, y_std
