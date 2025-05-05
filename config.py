import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

STOCK_LIST_PATH = os.path.join(DATA_DIR, 'stock_list.txt')
RAW_DIR         = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR   = os.path.join(DATA_DIR, 'processed')
LOG_DIR         = os.path.join(DATA_DIR, 'log')
CHART_DATA_DIR  = os.path.join(DATA_DIR, 'chart_data')

# Create a folder path for baseline assessment results
BASELINE_RESULT_DIR = os.path.join(DATA_DIR, 'baselines_evaluation_results')

# -----------------------------------------------------------------------------
# Neural-Net Hyperparameters
# -----------------------------------------------------------------------------
# All the parameters required by TorchClassifier / TorchRegressor are defined here
NN_ESTIMATOR_PARAMS = {
    # If you want input_dim to be automatically inferred, leave it as None
    'input_dim': None,

    # Hidden layer structure: can be freely increased, decreased, and resized
    'hidden_dims': (128, 64),

    # Optimizer related
    'lr': 1e-3,
    'batch_size': 256,

    # Training Cycle Related
    'epochs': 150,
    'patience': 15,

    # Device: 'cuda' / 'cpu', None means automatic detection
    'device': None,
}
