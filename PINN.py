import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from scipy.signal import find_peaks


# Custom loss function for the PINN
class CustomLossHC(nn.Module):
    def __init__(self, surrogate_model, half_cell_model, scaling_factors):
        super().__init__()
        self.surrogate_model = surrogate_model
        self.half_cell_model = half_cell_model
        self.scaling_factors = scaling_factors

    def forward(self, y_pred, y_true, y_true_dd, train_data):
        # Data-driven loss
        min_size = min(y_pred.shape[0], y_true_dd.shape[0])
        # Use the dynamically determined size for the loss calculation
        data_driven_loss = torch.mean((y_pred[:min_size] - y_true_dd[:min_size].cpu().double()) ** 2)

        # Physics-based loss
        y_pred_transformed = self.half_cell_model(y_pred.cpu().double())
        physics_based_loss = torch.mean((y_pred_transformed[:, 2:4] - y_true[:, 2:4].cpu().double()) ** 2)

        # dQ/dV peaks loss based on the surrogate model
        predicted_peaks = self.surrogate_model.predict(y_pred.detach().numpy())
        squared_diffs = []

        for i, experimental_dQdV in enumerate(train_data):
            peaks, _ = find_peaks(experimental_dQdV)
            sorted_peaks = sorted(peaks, key=lambda p: experimental_dQdV[p], reverse=True)
            x = np.linspace(3, 4.2, 100)

            if len(sorted_peaks) >= 2:
                peak_positions = [x[sorted_peaks[0]], x[sorted_peaks[1]]]
                squared_diff = (torch.tensor(peak_positions) - predicted_peaks[i]) ** 2
                squared_diffs.append(squared_diff)
            else:
                squared_diffs.append(torch.tensor([0.0, 0.0]))

        peak_loss = torch.mean(torch.cat(squared_diffs)) if squared_diffs else torch.tensor(0.0)

        # Total loss
        total_loss = (self.scaling_factors[0] * data_driven_loss +
                      self.scaling_factors[1] * physics_based_loss +
                      self.scaling_factors[2] * peak_loss)
        return total_loss


# PINN model architecture
class PINN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Training and evaluation pipeline
class PINNTrainer:
    def __init__(self, input_size, hidden1, hidden2, output_size, surrogate_model_path, half_cell_model_path,
                 learning_rate, batch_size, epochs, patience, scaling_factors, seed):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.scaling_factors = scaling_factors

        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Load models
        self.surrogate_model = joblib.load(surrogate_model_path)
        self.half_cell_model = joblib.load(half_cell_model_path)

        # Initialize the PINN model
        self.model = PINN(input_size, hidden1, hidden2, output_size)
        self.criterion = CustomLossHC(self.surrogate_model, self.half_cell_model, scaling_factors)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_data, train_labels, dd_labels, test_data, test_labels):
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        counter = 0

        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(inputs)
                loss = self.criterion(y_pred.cpu().double(), targets.cpu().double(), dd_labels, train_data.numpy())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validation
            with torch.no_grad():
                self.model.eval()
                y_pred_test = self.model(test_data)
                test_loss = self.criterion(y_pred_test, test_labels, dd_labels, test_data.numpy())
                self.model.train()

                if test_loss < best_loss:
                    best_loss = test_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.patience:
                        print(f"Stopping early at epoch {epoch + 1}.")
                        break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {running_loss / len(train_loader):.6f}, Test Loss = {test_loss:.6f}")


# Example usage
def main():
    # Assuming X_train, y_train, y_true_dd, X_test, and y_test are preloaded Tensors
    input_size = 100
    hidden1_neurons = 100
    hidden2_neurons = 10
    output_size = 4

    trainer = PINNTrainer(
        input_size=input_size,
        hidden1=hidden1_neurons,
        hidden2=hidden2_neurons,
        output_size=output_size,
        surrogate_model_path='surrogate_model.pkl',
        half_cell_model_path='half_cell_model.pkl',
        learning_rate=0.002,
        batch_size=100,
        epochs=5000,
        patience=500,
        scaling_factors=[1.0, 1.0, 1.0],
        seed=40
    )

    trainer.train(X_train, y_train, y_true_dd, X_test, y_test)


if __name__ == "__main__":
    main()
