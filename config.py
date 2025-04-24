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