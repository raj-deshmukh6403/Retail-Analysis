# recommendation_system.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

def create_recommendation_system(df, n_recommendations=5):
    """Create a product recommendation system using collaborative filtering"""
    print("Creating product recommendation system...")
    
    # Create a customer-item matrix (pivot table)
    customer_item_matrix = df.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    
    # Get product descriptions (for more informative recommendations)
    # Group by StockCode and take the first description to avoid Series objects later
    product_desc = df[['StockCode', 'Description']].drop_duplicates('StockCode')
    product_desc = product_desc.set_index('StockCode')
    
    # Convert the matrix to a sparse matrix for efficiency
    customer_item_sparse = csr_matrix(customer_item_matrix.values)
    
    # Fit the nearest neighbors model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(customer_item_sparse)
    
    # Generate recommendations for each customer
    all_recommendations = []
    
    # For each customer
    for customer_idx, customer_id in enumerate(customer_item_matrix.index):
        # Get the customer's purchases
        customer_purchases = customer_item_matrix.iloc[customer_idx, :].values.reshape(1, -1)
        
        # Find similar customers
        distances, indices = model.kneighbors(customer_purchases, n_neighbors=6)  # +1 because the first match is the customer itself
        
        # Skip the first match (the customer itself)
        similar_customers_indices = indices.flatten()[1:]
        
        # Get items purchased by similar customers but not by the current customer
        recommendations = set()
        
        for similar_idx in similar_customers_indices:
            # Get items purchased by similar customer
            similar_customer_id = customer_item_matrix.index[similar_idx]
            similar_purchases = customer_item_matrix.iloc[similar_idx, :]
            
            # Get items with quantities > 0 (items purchased)
            purchased_items = similar_purchases[similar_purchases > 0].index.tolist()
            
            # Get items not purchased by the current customer
            customer_purchased = customer_item_matrix.iloc[customer_idx, :]
            customer_purchased_items = customer_purchased[customer_purchased > 0].index.tolist()
            
            # Add new items to recommendations
            new_items = [item for item in purchased_items if item not in customer_purchased_items]
            recommendations.update(new_items)
            
            # If we have enough recommendations, stop
            if len(recommendations) >= n_recommendations:
                break
        
        # Get the top N recommendations (limit to n_recommendations)
        top_recommendations = list(recommendations)[:n_recommendations]
        
        # Add product descriptions for better readability
        for item in top_recommendations:
            # Get description if available - handle potentially returning a Series
            if item in product_desc.index:
                description = product_desc.loc[item, 'Description']
                # Ensure we get a single string, not a Series
                if isinstance(description, pd.Series):
                    description = description.iloc[0]  # Take first description if multiple exist
            else:
                description = 'Unknown'
            
            all_recommendations.append({
                'CustomerID': customer_id,
                'RecommendedStockCode': item,
                'ProductDescription': description
            })
    
    # Convert to dataframe
    recommendations_df = pd.DataFrame(all_recommendations)
    
    print(f"Generated {len(recommendations_df)} recommendations for {recommendations_df['CustomerID'].nunique()} customers")
    return recommendations_df

def save_recommendations(recommendations_df, output_path='data/processed/product_recommendations.csv'):
    """Save product recommendations"""
    # Ensure we're saving the full product descriptions without truncation
    pd.set_option('display.max_colwidth', None)
    recommendations_df.to_csv(output_path, index=False)
    # Reset display options
    pd.reset_option('display.max_colwidth')
    print(f"Product recommendations saved to {output_path}")

def visualize_top_recommended_products(recommendations_df):
    """Create visualization for most recommended products"""
    # Count occurrences of each product in recommendations
    product_counts = recommendations_df['RecommendedStockCode'].value_counts().reset_index()
    product_counts.columns = ['StockCode', 'RecommendationCount']
    
    # Add product descriptions - ensure we're working with string values
    product_descriptions = recommendations_df[['RecommendedStockCode', 'ProductDescription']].copy()
    
    # Convert any Series objects to strings
    if product_descriptions['ProductDescription'].apply(type).eq(pd.Series).any():
        product_descriptions['ProductDescription'] = product_descriptions['ProductDescription'].apply(
            lambda x: x.iloc[0] if isinstance(x, pd.Series) else x
        )
    
    # Now drop duplicates safely
    product_descriptions = product_descriptions.drop_duplicates()
    product_descriptions.columns = ['StockCode', 'Description']
    
    # Merge counts with descriptions
    top_products = pd.merge(product_counts, product_descriptions, on='StockCode')
    
    # Get top 10 products
    top_10 = top_products.head(10)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(x='RecommendationCount', y='Description', data=top_10)
    plt.title('Top 10 Most Recommended Products')
    plt.tight_layout()
    plt.savefig('visualizations/top_recommended_products.png')
    print("Top recommended products visualization saved to 'visualizations/top_recommended_products.png'")
    
    # Create customer recommendation distribution
    plt.figure(figsize=(10, 6))
    customer_rec_counts = recommendations_df.groupby('CustomerID').size()
    sns.histplot(customer_rec_counts, kde=True)
    plt.title('Distribution of Recommendations per Customer')
    plt.xlabel('Number of Recommendations')
    plt.ylabel('Number of Customers')
    plt.savefig('visualizations/recommendation_distribution.png')
    print("Recommendation distribution visualization saved to 'visualizations/recommendation_distribution.png'")

def main():
    from data_processing import load_data, clean_retail_data
    
    # Load and clean data
    df = clean_retail_data(load_data())
    
    # Create recommendation system
    recommendations_df = create_recommendation_system(df)
    
    # Save recommendations
    save_recommendations(recommendations_df)
    
    # Visualize top recommended products
    visualize_top_recommended_products(recommendations_df)
    
    return recommendations_df

if __name__ == "__main__":
    main()