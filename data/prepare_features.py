import os
import pandas as pd

from features.volume_features import add_volume_features
from features.technical_indicators import add_technical_indicators
from features.price_features import add_price_features

INPUT_DIR  = 'data/raw'
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)   # Make sure the output folder exists

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith('.csv'):
        continue

    input_path  = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    print(f"[INFO]Processing: {filename}")

    df = pd.read_csv(input_path)

    # Make sure have the necessary fields
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        print(f"[Warning]{filename} is missing a required field and has been skipped.")
        continue

    # ==== START: Processing ====
    df = add_volume_features(df)
    df = add_technical_indicators(df)
    df = add_price_features(df)
    # ==== END: Processing ======

    df.to_csv(output_path, index=False)
    print(f"[INFO]Saved to: {output_path}")
