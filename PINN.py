import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import joblib
from scipy.signal import find_peaks
import numpy as np

# Customizable parameters
surrogate_model_path = 'surrogate_model.pkl'  # Path to the surrogate model
half_cell_model_path = 'half_cell_model.pkl'  # Path to the half-cell model
batch_size = 100  # Customize as needed
learning_rate = 0.002

# Customizable architecture parameters
input_size = 100  # Size of the input layer
hidden1_neurons = 100  # Number of neurons in the first hidden layer
hidden2_neurons = 10  # Number of neurons in the second hidden layer
output_size = 4  # Size of the output layer

# Customizable training parameters
epochs = 5000  # Number of epochs for training
loss_threshold = 0.0005  # Loss threshold for early stopping
patience = 500  # Patience for early stopping
seed = 40  # Random seed for reproducibility

# Set up random seed for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Print initial seed for reproducibility
initial_seed = torch.initial_seed()
print("Initial Seed:", initial_seed)

# Load pretrained models
surrogate_model = joblib.load(surrogate_model_path)
half_cell_model = joblib.load(half_cell_model_path)

# Define custom loss function
class CustomLossHC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, y_true_dd, surrogate_model, half_cell_model, train_data):
        # Data-driven loss
        data_driven_loss = torch.mean((y_pred[0:180] - y_true_dd[0:180].cpu().double()) ** 2)

        # Physics-based loss
        y_pred_transformed = half_cell_model(y_pred.cpu().double())
        physics_based_loss = torch.mean((y_pred_transformed[:, 2:4] - y_true[:, 2:4].cpu().double()) ** 2)

        # dQdV peaks loss based on surrogate model
        predicted_peaks = surrogate_model.predict(y_pred.detach().numpy())
        squared_diffs = []
        
        for i, experimental_dQdV in enumerate(train_data):
            peaks, _ = find_peaks(experimental_dQdV, prominence=prominence, distance=distance)
            sorted_peaks = sorted(peaks, key=lambda p: experimental_dQdV[p], reverse=True)
            x = np.linspace(3, 4.2, 100)
            
            if len(sorted_peaks) >= 2:
                peak_positions = [x[sorted_peaks[0]], x[sorted_peaks[1]]]
                squared_diff = (torch.tensor(peak_positions) - predicted_peaks[i]) ** 2
                squared_diffs.append(squared_diff)
            else:
                squared_diffs.append(torch.tensor([0.0, 0.0]))

        if squared_diffs:
            peak_loss = torch.mean(torch.cat(squared_diffs))
        else:
            peak_loss = torch.tensor(0.0)

        total_loss = (scaling_factors[0] * data_driven_loss +
                      scaling_factors[1] * physics_based_loss +
                      scaling_factors[2] * peak_loss)
        
        return total_loss

# Define PINN model architecture
class PINN(nn.Module):
    def __init__(self, input_size=input_size, hidden1=hidden1_neurons, hidden2=hidden2_neurons, output_size=output_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load and normalize data for a general fold
X_train = torch.from_numpy(train_augmented_set.values).float()
y_train = torch.from_numpy(train_labels.values).float()
y_true_dd = torch.from_numpy(dd_labels.values).float()
X_test = torch.from_numpy(test_set.values).float()
y_test = torch.from_numpy(test_labels.values).float()

X_train = F.normalize(X_train, dim=0)
X_test = F.normalize(X_test, dim=0)

# Initialize model, loss function, optimizer
model = PINN(input_size=input_size, hidden1=hidden1_neurons, hidden2=hidden2_neurons, output_size=output_size)
criterion = CustomLossHC()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training with early stopping and thresholding
best_loss = float('inf')  # Initialize best loss
counter = 0  # Initialize counter for early stopping

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(inputs)
        loss = criterion(y_pred.cpu().double(), targets.cpu().double(), y_true_dd, surrogate_model, half_cell_model, X_train.numpy())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {running_loss / (i + 1):.6f}")

    with torch.no_grad():
        model.eval()
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test, y_true_dd, surrogate_model, half_cell_model, X_test.numpy())
        print(f"Epoch {epoch + 1}: test_loss = {test_loss:.6f}")
        model.train()

        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience or test_loss < loss_threshold:
                print(f"Stopping at epoch {epoch + 1}")
                break
