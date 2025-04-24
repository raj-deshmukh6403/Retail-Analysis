# main.py
import os
import pandas as pd
from data_processing import load_data, clean_retail_data, save_processed_data
from customer_segmentation import create_rfm_features, save_customer_segments, visualize_segments
from market_basket_analysis import create_basket_analysis, save_association_rules, visualize_top_associations
from purchase_prediction import create_target_variable, create_purchase_prediction_features, train_purchase_model, save_model_artifacts, create_predictions, visualize_model_performance
from recommendation_system import create_recommendation_system, save_recommendations, visualize_top_recommended_products

def ensure_directories_exist():
    """Ensure all required directories exist"""
    directories = ['data/raw', 'data/processed', 'models', 'visualizations', 'notebooks']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_pipeline():
    """Run the complete data processing and analysis pipeline"""
    print("Starting Retail Customer Analysis Project Pipeline")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Step 1: Data Processing
    print("\n--- Step 1: Data Processing ---")
    if os.path.exists('data/raw/Online_Retail.xlsx'):
        df = load_data('data/raw/Online_Retail.xlsx')
    else:
        print("Error: Data file not found. Please download Online_Retail.xlsx and place it in the data/raw directory.")
        return
    
    df_clean = clean_retail_data(df)
    save_processed_data(df_clean)
    
    # Step 2: Customer Segmentation
    print("\n--- Step 2: Customer Segmentation ---")
    rfm_df = create_rfm_features(df_clean)
    save_customer_segments(rfm_df)
    visualize_segments(rfm_df)
    
    # Step 3: Market Basket Analysis
    print("\n--- Step 3: Market Basket Analysis ---")
    association_rules_df, basket_sets = create_basket_analysis(df_clean)
    save_association_rules(association_rules_df)
    visualize_top_associations(association_rules_df)
    
    # Step 4: Purchase Prediction Model
    print("\n--- Step 4: Purchase Prediction Model ---")
    train_df, target_df = create_target_variable(df_clean)
    customer_features = create_purchase_prediction_features(train_df)
    model, scaler, feature_importance, X_test, y_test = train_purchase_model(customer_features, target_df)
    save_model_artifacts(model, scaler, feature_importance)
    prediction_df = create_predictions(df_clean, model, scaler)
    visualize_model_performance(model, X_test, y_test)
    
    # Step 5: Recommendation System
    print("\n--- Step 5: Recommendation System ---")
    recommendations_df = create_recommendation_system(df_clean)
    save_recommendations(recommendations_df)
    visualize_top_recommended_products(recommendations_df)
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print(f"Data files generated in 'data/processed/' directory:")
    for file in os.listdir('data/processed/'):
        print(f"  - {file}")
    print("\nYou can now import these files into Power BI to create your dashboard.")

if __name__ == "__main__":
    run_pipeline()