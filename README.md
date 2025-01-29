Main codes for half-cell model, PINN and co-kriging implemented for physics-informed degradation diagnostics project: https://doi.org/10.1016/j.ensm.2024.103343


![image](https://github.com/user-attachments/assets/44d05150-5d04-43ea-9423-c6a1544b2db6)



# PINN 
A **Physics-Informed Neural Network (PINN)** model designed for battery degradation diagnostics. This model combines:

### 1️⃣ **CustomLossHC Class**
Defines the **hybrid loss function**, including:
- **Data-driven MSE loss** (between predictions and real data).
- **Physics-based loss** (constraining outputs using a half-cell model).
- **Peak loss** (minimizing dQ/dV peak differences using a surrogate model).

### 2️⃣ **PINN Model**
Implements a **fully connected neural network (FCNN)** with:
- **Input layer** → `input_size` neurons.
- **Two hidden layers** with ReLU activation.
- **Output layer** → `output_size` neurons.

### 3️⃣ **PINNTrainer Class**
Handles the **training pipeline**, including:
- **Loading pre-trained surrogate and half-cell models**.
- **Training with Adam optimizer** and early stopping.
- **Batch-based training using PyTorch DataLoader**.
- **Validation to monitor performance**.



# Co-Kriging 
Co-Kriging is an extension of Gaussian Process Regression (GPR) that enables multi-fidelity modeling. This allows us to model a high-fidelity function using both high-fidelity (experimental) and low-fidelity (simulated by the half-cell model) datasets. This method utilizes a **joint covariance function** to simultaneously model the auto-covariances of each individual process and the **cross-covariance** between two related processes. The model is optimized jointly, which means that both the kernel parameters and the relationship between the two outputs are learned at the same time.

### 1️⃣ **CoKrigingModel Class**
The `CoKrigingModel` class handles **training, optimization, and prediction** using Co-Kriging.  

#### **Methods:**
- **`__init__()`** → Initializes the model, loads data, and defines the kernels.
- **`get_data_and_split()`** → Loads the dataset and splits it into training/testing sets.
- **`train_model()`** → Optimizes the Co-Kriging model using maximum likelihood estimation.
- **`predict()`** → Generates predictions for new test inputs.
- **`run()`** → Trains the model, makes predictions, and calculates **Root Mean Square Percentage Error (RMSPE)**.

### 2️⃣ **test_cokriging() Function**
This function serves as the main entry point to **train and evaluate** the Co-Kriging model.








