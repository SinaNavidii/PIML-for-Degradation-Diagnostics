Main codes for half-cell model, PINN and co-kriging implemented for physics-informed degradation diagnostics project: https://doi.org/10.1016/j.ensm.2024.103343


![image](https://github.com/user-attachments/assets/44d05150-5d04-43ea-9423-c6a1544b2db6)



# PINN model for battery degradation diagnostics

This repository provides a **Physics-Informed Neural Network (PINN)** model designed for battery degradation diagnostics. The model combines:

- **Data-Driven Loss**: A traditional mean squared error (MSE) loss between predicted and experimental data.
- **Physics-Based Loss**: A loss that enforces physical constraints by comparing model predictions to known battery behaviors.
- **Peak Loss**: A term that encourages the model to align predictions of capacity drop with experimental peak data using the surrogate model.

### Key Components:

- **Surrogate Model**: Pre-trained model (loaded using joblib) that predicts peak values.
- **Half-Cell Model**: Transforms predicted data into a format compatible with the physical constraints of battery operation.
- **PINN Model**: A neural network with two hidden layers and ReLU activations, used to predict battery degradation parameters based on input features.

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
- **Physics-based loss**: Ensures predictions align with physical models.
- **Peak loss**: Penalizes incorrect peak predictions.

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



## Co-Kriging 
This method utilizes a **joint covariance function** to simultaneously model the auto-covariances of each individual process and the **cross-covariance** between two related processes. The model is optimized jointly, which means that both the kernel parameters and the relationship between the two outputs are learned at the same time.

### Key Steps:
1. **Train a Joint Gaussian Process (GPCoregionalizedRegression):** 
   - This involves using both **auto-covariance kernels** for each individual process and a **cross-covariance kernel** to model the correlation between the two processes within a single GP model.
  
2. **Simultaneous Optimization:** 
   - The model is optimized by adjusting the parameters of the auto-covariances and the cross-covariance term together. This ensures that both the individual variances and the relationship between the processes are learned simultaneously.
   
3. **Final Prediction:** 
   - Once the model has been trained, it makes predictions using both processes, leveraging their joint structure to provide a more accurate and informed output. The final prediction is derived from the jointly learned processes.


## Dalta learning with GP (Co-kriging with single correlation parameter)

In this approach, the two GP models are trained separately, and the correlation between them is controlled by the **scalar parameter `rho`**. This parameter adjusts the derived output based on the primary GP predictions.

### Steps:
1. **Train Two Separate GPs:** One for the primary output and one for the derived output.
2. **Optimization of `rho`:** `rho` is optimized to adjust the predictions of the derived output based on the primary GP's predictions.
3. **Final Prediction:** The final prediction is a combination of the predictions from both models, adjusted by `rho`.

#### Objective Function for Optimization:
- `rho` is optimized by minimizing the negative log-likelihood of the second GP model, which uses the residuals between the primary GP's predictions and the true derived output.


