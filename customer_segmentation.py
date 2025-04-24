# customer_segmentation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def create_rfm_features(df, end_date=None):
    """Create RFM features for customer segmentation"""
    print("Creating RFM features...")
    
    if end_date is None:
        end_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (end_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalAmount': 'sum'  # Monetary
    })
    
    # Rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Create RFM segments
    rfm['R_Quartile'] = pd.qcut(rfm['Recency'], 4, labels=False, duplicates='drop')
    rfm['F_Quartile'] = pd.qcut(rfm['Frequency'], 4, labels=False, duplicates='drop')
    rfm['M_Quartile'] = pd.qcut(rfm['Monetary'], 4, labels=False, duplicates='drop')
    
    # Invert recency quartile (lower values are better)
    rfm['R_Quartile'] = 3 - rfm['R_Quartile']
    
    # Create RFM score
    rfm['RFM_Score'] = rfm['R_Quartile'].astype(str) + rfm['F_Quartile'].astype(str) + rfm['M_Quartile'].astype(str)
    
    # Create segment labels
    segment_map = {
        '333': 'Champions',
        '332': 'Loyal Customers',
        '323': 'Potential Loyalists',
        '322': 'Promising',
        '331': 'Recent Customers',
        '321': 'Customers Needing Attention',
        '311': 'About To Sleep',
        '222': 'Need Awakening',
        '223': 'At Risk',
        '221': 'Can\'t Lose Them',
        '213': 'Hibernating',
        '211': 'Lost',
        '111': 'Lost Cheap Customers'
    }
    
    # Apply mapping for selected combinations
    rfm['Segment'] = rfm['RFM_Score'].map(lambda x: segment_map.get(x, 'Other'))
    
    print(f"RFM segmentation created for {rfm.shape[0]} customers")
    return rfm

def save_customer_segments(rfm_df, output_path='data/processed/customer_segments.csv'):
    """Save customer segmentation data"""
    rfm_df.reset_index().to_csv(output_path, index=False)
    print(f"Customer segments saved to {output_path}")

def visualize_segments(rfm_df):
    """Create basic visualizations of customer segments"""
    plt.figure(figsize=(10, 6))
    segment_counts = rfm_df['Segment'].value_counts()
    sns.barplot(x=segment_counts.index, y=segment_counts.values)
    plt.title('Customer Segments Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('visualizations/customer_segments.png')
    print("Customer segment visualization saved to 'visualizations/customer_segments.png'")

def main(df=None):
    if df is None:
        from data_processing import load_data, clean_retail_data
        df = clean_retail_data(load_data())
    
    # Create RFM features
    rfm_df = create_rfm_features(df)
    
    # Save customer segments
    save_customer_segments(rfm_df)
    
    # Create visualization
    visualize_segments(rfm_df)
    
    return rfm_df

if __name__ == "__main__":
    main()