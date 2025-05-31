# Physics-Informed Machine Learning for Battery Degradation Diagnostics  
Main codes for half-cell model, PINN and co-kriging implemented for physics-informed degradation diagnostics project: https://doi.org/10.1016/j.ensm.2024.103343

![image](https://github.com/user-attachments/assets/44d05150-5d04-43ea-9423-c6a1544b2db6)

<sub>**Note:** The dataset used in this project is **confidential** and **cannot be shared** due to agreements with industry collaborators.</sub>

# PINN 
A **physics-informed neural network (PINN)** model designed for battery degradation diagnostics. This model combines:

### 1️⃣ **CustomLossHC Class**
Defines the **hybrid loss function**, including:
- **Data-driven MSE loss** (between predicted and true values).
- **Physics-based loss** (constraining outputs using a half-cell model).
- **Peak loss** (minimizing dQ/dV peak differences using a surrogate model).

### 2️⃣ **PINN Model**
Implements a **fully connected neural network** with:
- **Input layer** → `input_size` neurons.
- **Two hidden layers** with ReLU activation.
- **Output layer** → `output_size` neurons.

### 3️⃣ **PINNTrainer Class**
Handles the **training pipeline**, including:
- **Loading pre-trained surrogate and half-cell models**.
- **Training with Adam optimizer** and early stopping.
- **Batch-based training using PyTorch DataLoader**.
- **Validation to monitor performance**.

![image](https://github.com/user-attachments/assets/ff5bad53-2e64-4db1-8b47-3a5b6695c8c8)




# Co-Kriging 
Co-kriging is an extension of Gaussian process regression (GPR) that enables multi-fidelity modeling. This allows us to model a high-fidelity function using both high-fidelity (experimental) and low-fidelity (simulated by the half-cell model) datasets. It assumes a linear relationship between the two fidelities, combining their predictions through a scaling parameter ρ and a discrepancy GP that captures the remaining difference.

### 1️⃣ **CoKrigingModel Class**
The `CoKrigingModel` class handles **training, optimization, and prediction**.  

#### **Methods:**
- **`__init__()`** → Initializes the model and loads separate low-fidelity and high-fidelity datasets.
- **`_load_data()`** → Loads and splits the dataset into LF training, HF training, and test sets.
- **`_fit_output()`** → Sequentially trains a LF GP, computes the discrepancy, and trains a second GP on the discrepancy.
- **`_build_covariance()`** → Constructs the full block covariance matrix across both fidelities.
- **`_log_marginal_likelihood()`** → Computes the log-marginal likelihood used to optimize ρ.
- **`train_model()`** → Optimizes ρ and fits both GPs for each output using maximum likelihood estimation.
- **`predict()`** → Generates predictions with posterior mean and variance for new test inputs.
- **`run()`** → Trains the model, makes predictions, and calculates RMSPE.

### 2️⃣ **test_cokriging() Function**
This function serves as the main entry point to **train and evaluate** the co-kriging model.












