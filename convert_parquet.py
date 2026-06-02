import os
import sys
import pandas as pd

def convert_parquet_to_excel(parquet_path):
    """
    Converts a Parquet file to an Excel (.xlsx) file.
    Takes the full path to the Parquet file.
    """
    # Check if the file actually exists
    if not os.path.exists(parquet_path):
        print(f"Error: The file at '{parquet_path}' does not exist.")
        return

    try:
        print("Reading Parquet file...")
        df = pd.read_parquet(parquet_path)
        
        # Create output path with the same name but .xlsx extension
        base_path, _ = os.path.splitext(parquet_path)
        excel_path = base_path + ".xlsx"
        
        print(f"Converting {len(df)} rows to Excel. Please wait...")
        # openpyxl engine is required for modern Excel (.xlsx) files
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        print(f"Success! Visible Excel file saved to:\n{excel_path}")
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    # Option A: If a path was passed via terminal argument
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    # Option B: Prompt the user to enter the path manually
    else:
        target_path = input("Please enter the full path to your Parquet file: ").strip()
        # Remove surrounding quotes if user dragged and dropped the file into terminal
        target_path = target_path.strip("'\"")
        
    convert_parquet_to_excel(target_path)