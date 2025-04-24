# data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(file_path='data/raw/Online Retail.xlsx'):
    """Load the retail dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def clean_retail_data(df):
    """Clean and prepare the retail dataset"""
    print("Cleaning data...")
    # Remove rows with missing customer IDs
    df = df.dropna(subset=['CustomerID'])
    
    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Convert invoice date to datetime format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Create a total amount column
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Remove returns (negative quantities) for initial analysis
    df_sales = df[df['Quantity'] > 0]
    
    # Remove cancelled orders
    df_sales = df_sales[~df_sales['InvoiceNo'].astype(str).str.contains('C')]
    
    print(f"Data cleaned: {df_sales.shape[0]} valid transactions from {df_sales['CustomerID'].nunique()} customers")
    return df_sales

def save_processed_data(df, output_path='data/processed/cleaned_retail_data.csv'):
    """Save the processed data"""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    # Load data
    df = load_data()
    
    # Clean data
    df_clean = clean_retail_data(df)
    
    # Save processed data
    save_processed_data(df_clean)
    
    return df_clean

if __name__ == "__main__":
    main()