"""
Script to convert Stata .dta file to Parquet format
Run this locally to create the parquet file, then commit it to git
"""
import pandas as pd
import os

# Path to the Stata file
stata_path = "./6389799/RORE_QJE_replication_v2/data/rore_public_main.dta"
parquet_path = "./6389799/RORE_QJE_replication_v2/data/rore_public_main.parquet"

print(f"Reading Stata file from: {stata_path}")

try:
    # Use pandas to read Stata file (most reliable)
    with open(stata_path, 'rb') as f:
        df = pd.read_stata(f)
    print("Successfully read with pandas")
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    
    # Save as Parquet
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"\nSuccessfully converted to Parquet: {parquet_path}")
    print(f"Parquet file size: {os.path.getsize(parquet_path) / 1024 / 1024:.2f} MB")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
