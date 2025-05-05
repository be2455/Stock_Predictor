# sklearn estimators made in PyTorch (Packaging sklearn interface with PyTorch)

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import torch, torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler

# === Project‑level imports ===================================================
from models.neural_net import Regression_Classification_NN

# =============================================================================
# BASE ESTIMATOR: Unified PyTorch Estimator (shared by Regressor / Classifier)
# =============================================================================
class _BaseTorchEstimator(BaseEstimator):
    def __init__(
        self,
        task: str,  # "reg"  -> regression  (MSE loss)
                    # "cls"  -> classification (BCE‑with‑logits)
        input_dim: int | None = None,   # Number of input features.
                                        # If None, inferred from X.shape[1] at fit‑time.
        hidden_dims: tuple[int, ...] = (128, 64),   # Sizes of hidden layers (MLP).
                                                    # Example: (128, 64) → 2 layers.
        lr: float = 1e-3,           # Learning rate for the Adam optimiser.
        epochs: int = 150,          # Maximum number of training epochs.
        batch_size: int = 256,      # Mini‑batch size for SGD.
        patience: int = 15,         # Early‑stopping patience (epochs without validation‑loss improvement before terminating training).
        device: str | None = None,  # `'cuda'`, `'cpu'`, or None.
                                    # None → automatically choose `'cuda'` if a GPU is available, else `'cpu'`.
    ):
        # Store hyper‑parameters
        self.task         = task
        self.input_dim    = input_dim
        self.hidden_dims  = hidden_dims
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.patience     = patience

        # Select compute device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # StandardScaler shared across train/val/test splits
        self.scaler_: StandardScaler = StandardScaler()

    # -----------------------------------------------------------------------------
    # TRAINING LOOP (sklearn-compatible .fit method)
    # -----------------------------------------------------------------------------
    def fit(self, X: NDArray[float], y: NDArray[float]) -> "_BaseTorchEstimator":
        " Replace the traditional scikit-learn `fit` workflow with a custom PyTorch training loop."

        # Infer input dimension if not provided during initialization
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        # 1. Scale features to zero mean and unit variance (sklearn StandardScaler)
        Xs = self.scaler_.fit_transform(X)
        # 2. Convert numpy array to PyTorch tensor and move to device (CPU/GPU)
        Xt = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        # 3. Prepare target tensor with shape (N, 1)
        yt = torch.tensor(y,  dtype=torch.float32).reshape(-1, 1).to(self.device)

        # Initialize the underlying MLP model and move it to the compute device
        self.model_ = Regression_Classification_NN(
            self.input_dim, list(self.hidden_dims)
        ).to(self.device)

        # Choose loss function based on task type: MSE for regression, BCEWithLogits for classification
        crit = nn.MSELoss() if self.task == "reg" else nn.BCEWithLogitsLoss()
        # Instantiate Adam optimizer with the specified learning rate
        opt  = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        # Variables for early stopping
        # best = None
            # 用來暫存「目前找到的最優模型權重」──等到有更好的驗證損失時，才會把模型的 state_dict() 複製到這個變數裡
        # best_loss = float('inf')
            # 代表「目前的最佳（最低）驗證損失」，一開始設為正無窮大，保證任何真實計算出來的 val_loss 都會比它小，第一個 epoch 的結果一定能把它更新
        # stall = 0
            # 計算「連續多少個 epoch 驗證損失沒進步」。每次 loss 沒下降就 stall += 1，只要 stall 累積到超過 patience（耐心值），訓練就提前停止
        best, best_loss, stall = None, float('inf'), 0

        # Training loop over epochs
        for _ in range(self.epochs):
            self.model_.train()            # Set model to training mode (enables dropout, batchnorm updates, etc.)
            idx = torch.randperm(len(Xt))  # Shuffle the data indices for stochastic mini-batch sampling

            # Mini-batch training
            for i in range(0, len(Xt), self.batch_size):
                b = idx[i:i+self.batch_size]
                # 1. Forward pass: compute model outputs
                out_reg, out_logit = self.model_(Xt[b])
                pred = out_reg if self.task == "reg" else out_logit
                # 2. Compute loss between predictions and targets
                loss = crit(pred, yt[b])
                # 3. Backward pass: zero gradients, compute new gradients
                opt.zero_grad()
                loss.backward()
                # 4. Parameter update: apply optimizer step
                opt.step()

            # ── Simplified validation on the entire training set ──
            self.model_.eval()  # Switch to evaluation mode
                                # Disable Dropout
                                # BatchNorm no longer updates running stats
            with torch.no_grad():
                # Turn off gradient calculation:
                    # Reduce memory usage
                    # Speed ​​up, because no intermediate gradients need to be calculated or stored
                out_reg, out_logit = self.model_(Xt)
                # Perform a forward pass on the entire training set (Xt) to obtain two sets of outputs: regression and classification.
                val_pred = out_reg if self.task == "reg" else out_logit
                # Select the corresponding output according to the task type and calculate the loss
                    # .item() converts the tensor to pure Python float
                val_loss = crit(val_pred, yt).item()

            # Early stopping logic: save best model or increment stall counter
            if val_loss < best_loss:
                best_loss, stall = val_loss, 0
                # Deep copy model state to CPU for later restoration
                best = {k: v.cpu() for k, v in self.model_.state_dict().items()}
            else:
                stall += 1

            # Break training if no improvement for 'patience' epochs
            if stall >= self.patience: break

        # Restore best model weights and set to evaluation mode
        self.model_.load_state_dict(best)        # best weights
        self.model_.eval()
        return self

    # -----------------------------------------------------------------------------
    # INFERENCE (used by predict/predict_proba, during validation/testing)
    # -----------------------------------------------------------------------------
    def _forward(self, X: NDArray[float]) -> Tuple[NDArray[float], NDArray[float]]:
        """
        Internal inference helper (used during validation/testing):
        1. Applies the fitted StandardScaler to trained data.
        2. Converts the scaled array into a PyTorch tensor on the correct device.
        3. Runs a forward pass without gradient tracking.
        4. Returns both regression outputs and classification logits as NumPy arrays.

        Returns:
            A tuple of two NumPy arrays:
            - out_reg: regression predictions, shape (n_samples, 1)
            - out_logit: raw classification logits, shape (n_samples, 1)
        """

        # Scale features using the existing scaler (zero mean, unit variance)
        Xs = self.scaler_.transform(X)
        # Convert the scaled data to a float32 tensor and move it to the target device (CPU or GPU)
        Xt = torch.tensor(Xs, dtype=torch.float32).to(self.device)

        # Disable gradient computation for inference to save memory and speed up
        with torch.no_grad():
            # Call the underlying PyTorch model's forward() to get both heads
            out_reg, out_logit = self.model_(Xt)

        # Move outputs back to CPU and convert to NumPy for compatibility with sklearn-style APIs
        return out_reg.cpu().numpy(), out_logit.cpu().numpy()

# =============================================================================
# REGRESSOR WRAPPER (for sklearn compatibility)
# =============================================================================
class TorchRegressor(_BaseTorchEstimator, RegressorMixin):
    def __init__(self, **kw) -> None:
        super().__init__(task="reg", **kw)

    def predict(self, X: NDArray[float]) -> NDArray[float]:
        reg, _ = self._forward(X)
        return reg.ravel()  # returns continuous values

# =============================================================================
# CLASSIFIER WRAPPER (for sklearn compatibility)
# =============================================================================
class TorchClassifier(_BaseTorchEstimator, ClassifierMixin):
    def __init__(self, **kw) -> None:
        super().__init__(task="cls", **kw)

    def predict(self, X: NDArray[float]) -> NDArray[int]:
        _, logit = self._forward(X)
        return (logit.ravel()>0).astype(int)  # returns hard labels

    def predict_proba(self, X: NDArray[float]) -> NDArray[float]:
        _, logit = self._forward(X)
        p = 1/(1+np.exp(-logit))
        return np.c_[1-p, p]  # returns probability for class 0 and 1
