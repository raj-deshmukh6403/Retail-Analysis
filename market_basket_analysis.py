# market_basket_analysis.py
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

def create_basket_analysis(df, min_support=0.01, min_confidence=0.2, max_items=1000):
    """Create market basket analysis using the Apriori algorithm with memory optimization"""
    print("Creating market basket analysis...")
    
    # Focus on most popular items to reduce memory usage
    popular_items = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(max_items).index
    print(f"Limiting analysis to top {len(popular_items)} most popular products to conserve memory")
    
    # Filter dataset to only include popular items
    df_filtered = df[df['Description'].isin(popular_items)].copy()
    
    # Group items by invoice
    # Create a sparse representation for efficiency
    basket = pd.crosstab(df_filtered['InvoiceNo'], df_filtered['Description'])
    
    # Convert to binary representation (1 = item purchased, 0 = item not purchased)
    basket_sets = basket.astype(bool).astype(int)
    
    # Remove columns with all zeros (items never purchased)
    basket_sets = basket_sets.loc[:, basket_sets.sum() > 0]
    
    # Apply apriori algorithm to find frequent itemsets
    print(f"Finding frequent itemsets with minimum support of {min_support}...")
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True, low_memory=True)
    
    # Check if any frequent itemsets were found
    if frequent_itemsets.empty:
        print("No frequent itemsets found with the current support threshold. Trying with a lower threshold...")
        min_support = min_support / 2
        frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True, low_memory=True)
    
    if frequent_itemsets.empty:
        print("Still no frequent itemsets found. Creating a placeholder output.")
        rules = pd.DataFrame(columns=['antecedents', 'consequents', 'antecedent support', 
                                      'consequent support', 'support', 'confidence', 'lift', 'leverage', 'conviction'])
    else:
        # Generate association rules
        print(f"Generating association rules with minimum confidence of {min_confidence}...")
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Sort rules by lift
        rules = rules.sort_values('lift', ascending=False)
        
        # Clean up the rules dataframe for better readability
        # Convert frozensets to strings to avoid truncation issues in CSV exports
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    print(f"Found {len(rules)} association rules")
    return rules, basket_sets

def save_association_rules(rules_df, output_path='data/processed/association_rules.csv'):
    """Save association rules to a CSV file"""
    # Make sure we're saving all columns without truncation
    pd.set_option('display.max_colwidth', None)
    
    # Check if dataframe is empty
    if rules_df.empty:
        # Create a sample rule to avoid empty file
        sample_rule = pd.DataFrame({
            'antecedents': ['No rules found'],
            'consequents': ['No rules found'],
            'antecedent support': [0],
            'consequent support': [0],
            'support': [0],
            'confidence': [0],
            'lift': [0],
            'leverage': [0],
            'conviction': [0]
        })
        sample_rule.to_csv(output_path, index=False, float_format='%.4f')
    else:
        rules_df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"Association rules saved to {output_path}")
    # Reset display options
    pd.reset_option('display.max_colwidth')

def visualize_top_associations(rules_df, n=10):
    """Create visualizations for top association rules"""
    plt.figure(figsize=(10, 8))
    
    # Check if we have enough rules
    if len(rules_df) == 0 or 'antecedents' not in rules_df.columns:
        plt.text(0.5, 0.5, 'No association rules found with current thresholds', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('Market Basket Analysis')
    else:
        # Get top N rules by lift (or all if we have fewer than N)
        top_n = min(n, len(rules_df))
        top_rules = rules_df.head(top_n)
        
        # Create rule labels
        labels = [f"{ant} → {con}" for ant, con in zip(top_rules['antecedents'], top_rules['consequents'])]
        
        # Plot
        if top_n > 0:
            sns.barplot(x=top_rules['lift'], y=labels)
            plt.title(f'Top {top_n} Product Associations by Lift')
            plt.xlabel('Lift')
        else:
            plt.text(0.5, 0.5, 'No significant product associations found', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Market Basket Analysis')
    
    plt.tight_layout()
    plt.savefig('visualizations/top_associations.png')
    print(f"Top associations visualization saved to 'visualizations/top_associations.png'")
    
    # Create a scatterplot of support vs confidence if we have rules
    plt.figure(figsize=(10, 8))
    
    if len(rules_df) > 1 and 'support' in rules_df.columns:
        plt.scatter(rules_df['support'], rules_df['confidence'], alpha=0.5, s=rules_df['lift']*20)
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Support vs Confidence for Association Rules')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for support-confidence visualization', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('Support vs Confidence')
    
    plt.tight_layout()
    plt.savefig('visualizations/support_confidence_scatter.png')
    print(f"Support-confidence visualization saved to 'visualizations/support_confidence_scatter.png'")

def main():
    from data_processing import load_data, clean_retail_data
    
    # Load and clean data
    df = clean_retail_data(load_data())
    
    # Create basket analysis
    rules_df, basket_sets = create_basket_analysis(df)
    
    # Save association rules
    save_association_rules(rules_df)
    
    # Visualize top associations
    visualize_top_associations(rules_df)
    
    return rules_df

if __name__ == "__main__":
    main()