Main codes for half-cell model, PINN and co-kriging implemented for physics-informed degradation diagnostics project: https://doi.org/10.1016/j.ensm.2024.103343


![image](https://github.com/user-attachments/assets/44d05150-5d04-43ea-9423-c6a1544b2db6)



# PINN model for battery degradation diagnostics

A **Physics-Informed Neural Network (PINN)** model designed for battery degradation diagnostics. This model combines:

- **Data-Driven Loss**: The standard data-driven loss is calculated using the predicted half-cell
model parameters and their corresponding true values obtained from early-life experimental and late-life simulated data. 
- **Physics-Based Loss**: The second loss term is used to measure the difference between the predicted capacity and lithium inventory degradation parameters obtained by passing the predicted half-cell model parameters from the network into the half-cell surrogate model and the true ones. 
- **Peak Loss**: The third loss term is the difference between the peak positions (voltages) observed in the simulated and experimental
dQ/dV (V) curves.

## Classes

### 1. CustomLossHC
Defines the custom loss function combining:
- **Data-driven loss**: MSE between predicted and true data.
- **Physics-based loss**: Ensures predictions align with the half-cell model (influencing capacity and lithium inventory predictions).
- **Peak loss**: Penalizes incorrect peak predictions (influencing positive and negative active mass predistions).

### 2. PINN
The core model with two hidden layers and ReLU activations.

#### Parameters:
- `input_size`, `hidden1`, `hidden2`, `output_size`: Model architecture.

### 3. PINNTrainer
Handles the training process using the custom loss.

#### Parameters:
- Hyperparameters: `input_size`, `hidden1`, `hidden2`, `output_size`, `learning_rate`, `batch_size`, `epochs`, `scaling_factors`, etc.

## Usage

### 1. Prepare Dataset
Ensure the dataset is in the correct format:
- **Input features**: `train_data`, `test_data`
- **Labels**: `train_labels`, `test_labels`
- **True labels for peak detection**: `dd_labels`

### 2. Set Hyperparameters
Customize the following parameters:
- `input_size`, `hidden1`, `hidden2`, `output_size`, `learning_rate`, etc.


### 3. Train the Model
Once your data and hyperparameters are ready, you can train the model by running the test_PINN.



# Co-Kriging 
Co-Kriging is an extension of Gaussian Process Regression (GPR) that enables multi-fidelity modeling. This allows us to model a high-fidelity function using both high-fidelity and low-fidelity datasets, improving prediction accuracy where high-fidelity data is limited. This method utilizes a **joint covariance function** to simultaneously model the auto-covariances of each individual process and the **cross-covariance** between two related processes. The model is optimized jointly, which means that both the kernel parameters and the relationship between the two outputs are learned at the same time.

### 1. **CoKrigingModel Class**
The `CoKrigingModel` class handles **training, optimization, and prediction** using Co-Kriging.  

#### **Methods:**
- **`__init__()`** → Initializes the model, loads data, and defines the kernels.
- **`get_data_and_split()`** → Loads the dataset and splits it into training/testing sets.
- **`train_model()`** → Optimizes the Co-Kriging model using maximum likelihood estimation.
- **`predict()`** → Generates predictions for new test inputs.
- **`run()`** → Trains the model, makes predictions, and calculates **Root Mean Square Percentage Error (RMSPE)**.

### 2. **test_cokriging() Function**
This function serves as the main entry point to **train and evaluate** the Co-Kriging model.








