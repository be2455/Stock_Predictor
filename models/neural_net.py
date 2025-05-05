import torch
import torch.nn as nn

class Regression_Classification_NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64], dropout: float = 0.3):
        """
        Neural network with multiple hidden layers and two output heads:
        - Regression head: single output for predicted return (no activation for linear output).
        - Classification head: single output for predicted probability (using sigmoid in evaluation).
        """
        super(Regression_Classification_NN, self).__init__()
        layers = []
        prev_dim = input_dim
        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.shared_network = nn.Sequential(*layers)
        # Regression head
        self.reg_head = nn.Linear(prev_dim, 1)
        # Classification head
        self.class_head = nn.Linear(prev_dim, 1)
        # Note: We'll apply Sigmoid to class_head output during evaluation or use BCEWithLogits for training.

    def forward(self, x: torch.Tensor):
        x = self.shared_network(x)
        reg_out = self.reg_head(x)        # Regression output (continuous)
        class_logit = self.class_head(x)  # Classification output (logits for binary classification)
        return reg_out, class_logit
