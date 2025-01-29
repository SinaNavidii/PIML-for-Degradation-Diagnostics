Main codes for half-cell model, PINN and co-kriging implemented for physics-informed degradation diagnostics project: https://doi.org/10.1016/j.ensm.2024.103343


![image](https://github.com/user-attachments/assets/44d05150-5d04-43ea-9423-c6a1544b2db6)



# PINN model for battery degradation diagnostics

A **Physics-Informed Neural Network (PINN)** model designed for battery degradation diagnostics. This model combines:

- **Data-Driven Loss**: The standard data-driven loss is calculated using the predicted half-cell
model parameters and their corresponding true values obtained from early-life experimental and latelife simulated data. 
- **Physics-Based Loss**: a second loss term is generated to measure the difference between the predicted capacity and lithium inventory degradation parameter obtained by passing the predicted half-cell model parameters from the network into the half-cell surrogate model and the true ones. 
- **Peak Loss**: the third loss term is the difference between the peak positions (voltages) observed in the simulated and experimental
dQ/dV (V) curves.

## Requirements

To run the code, you will need to install the following dependencies:

- `torch`
- `numpy`
- `scikit-learn`
- `joblib`
- `scipy`

## Classes

### 1. CustomLossHC
Defines the custom loss function combining:
- **Data-driven loss**: MSE between predicted and true data.
- **Physics-based loss**: Ensures predictions align with the half-cell model (influencing capacity and lithium inventory predictions).
- **Peak loss**: Penalizes incorrect peak predictions (influencing positive and negative active mass predistions).

### 2. PINN
The core model with three fully connected layers and ReLU activations.

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
Modify the following parameters:
- `input_size`, `hidden1`, `hidden2`, `output_size`, `learning_rate`, etc.


### 3. Train the Model
Once your data and hyperparameters are ready, you can train the model by running the test_PINN.



# Co-Kriging  
This method utilizes a **joint covariance function** to simultaneously model the auto-covariances of each individual process and the **cross-covariance** between two related processes. The model is optimized jointly, which means that both the kernel parameters and the relationship between the two outputs are learned at the same time.

### Options for Co-Kriging:
1. **Using a scalar parameter** to adjust the correlation between two separate GP models.
2. **Using joint covariance structure**, where both **auto-covariance** and **cross-covariance** terms are optimized simultaneously.

### Key Steps:
1. **Train a Joint Gaussian Process (GPCoregionalizedRegression):** 
   - This involves using both **auto-covariance kernels** for each individual process and a **cross-covariance kernel** to model the correlation between the two processes within a single GP model.
  
2. **Simultaneous Optimization:** 
   - The model is optimized by adjusting the parameters of the auto-covariances and the cross-covariance term together. This ensures that both the individual variances and the relationship between the processes are learned simultaneously.
   
3. **Final Prediction:** 
   - Once the model has been trained, it makes predictions using both processes, leveraging their joint structure to provide a more accurate and informed output. The final prediction is derived from the jointly learned processes.





