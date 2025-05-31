# Physics-Informed Machine Learning for Battery Degradation Diagnostics  
Main codes for half-cell model, PINN and co-kriging implemented for physics-informed degradation diagnostics project: https://doi.org/10.1016/j.ensm.2024.103343

![image](https://github.com/user-attachments/assets/44d05150-5d04-43ea-9423-c6a1544b2db6)

<sub>**Note:** The dataset used in this project is **confidential** and **cannot be shared** due to agreements with industry collaborators.</sub>

# PINN 
A **physics-informed neural network (PINN)** model designed for battery degradation diagnostics. This model combines:

### 1️⃣ **CustomLossHC Class**
Defines the **hybrid loss function**, including:
- **Data-driven MSE loss** (between predictions and real data).
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
Co-kriging is an extension of Gaussian process regression (GPR) that enables multi-fidelity modeling. This allows us to model a high-fidelity function using both high-fidelity (experimental) and low-fidelity (simulated by the half-cell model) datasets. This method utilizes a **joint covariance function** to simultaneously model the auto-covariances of each individual process and the **cross-covariance** between two related processes. 

### 1️⃣ **CoKrigingModel Class**
The `CoKrigingModel` class handles **training, optimization, and prediction**.  

#### **Methods:**
- **`__init__()`** → Initializes the model, loads data, and defines the kernels.
- **`get_data_and_split()`** → Loads the dataset and splits it into training/testing sets.
- **`train_model()`** → Optimizes the Co-Kriging model using maximum likelihood estimation.
- **`predict()`** → Generates predictions for new test inputs.
- **`run()`** → Trains the model, makes predictions, and calculates RMSPE.

### 2️⃣ **test_cokriging() Function**
This function serves as the main entry point to **train and evaluate** the co-kriging model.












