# purchase_prediction.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def create_target_variable(df, future_days=30):
    """Create a target variable based on future purchases"""
    print("Creating target variable for purchase prediction...")
    
    # Get the maximum date in the dataset
    max_date = df['InvoiceDate'].max()
    
    # Define a cutoff date (30 days before max date)
    cutoff_date = max_date - timedelta(days=future_days)
    
    # Split data into training and future periods
    train_df = df[df['InvoiceDate'] <= cutoff_date].copy()
    future_df = df[df['InvoiceDate'] > cutoff_date].copy()
    
    # Create target variable: Did customer purchase in the future period?
    future_customers = future_df['CustomerID'].unique()
    
    # Create a dataframe with all customers
    all_customers = pd.DataFrame({'CustomerID': df['CustomerID'].unique()})
    
    # Create target variable (1 if customer made a purchase in future period, 0 otherwise)
    all_customers['Target'] = all_customers['CustomerID'].apply(lambda x: 1 if x in future_customers else 0)
    
    print(f"Created target variable with {len(all_customers)} customers")
    print(f"Positive class: {sum(all_customers['Target'])} customers ({sum(all_customers['Target'])/len(all_customers)*100:.2f}%)")
    
    return train_df, all_customers

def create_purchase_prediction_features(train_df):
    """Create features for purchase prediction model"""
    print("Creating features for purchase prediction model...")
    
    # Group by customer
    customer_features = train_df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',                  # Number of orders
        'InvoiceDate': [
            lambda x: (x.max() - x.min()).days,  # Customer tenure in days
            'max'                                # Date of last purchase
        ],
        'StockCode': 'nunique',                  # Number of unique products purchased
        'Quantity': ['sum', 'mean', 'std'],      # Quantity statistics
        'TotalAmount': ['sum', 'mean', 'std']    # Monetary statistics
    })
    
    # Flatten column hierarchy
    customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
    
    # Rename columns for clarity
    customer_features.rename(columns={
        'InvoiceNo_nunique': 'PurchaseFrequency',
        'InvoiceDate_<lambda>': 'CustomerTenure',
        'InvoiceDate_max': 'LastPurchaseDate',
        'StockCode_nunique': 'UniqueProductCount',
        'Quantity_sum': 'TotalQuantity',
        'Quantity_mean': 'AvgQuantity',
        'Quantity_std': 'StdQuantity',
        'TotalAmount_sum': 'TotalSpent',
        'TotalAmount_mean': 'AvgOrderValue',
        'TotalAmount_std': 'StdOrderValue'
    }, inplace=True)
    
    # Calculate days since last purchase
    max_date = customer_features['LastPurchaseDate'].max()
    customer_features['DaysSinceLastPurchase'] = (max_date - customer_features['LastPurchaseDate']).dt.days
    
    # Drop date column as it's not needed for modeling
    customer_features.drop(columns=['LastPurchaseDate'], inplace=True)
    
    # Handle missing values
    customer_features.fillna(0, inplace=True)
    
    # Reset index to make CustomerID a column
    customer_features.reset_index(inplace=True)
    
    # Save the features
    customer_features.to_csv('data/processed/customer_purchase_features.csv', index=False)
    print(f"Customer features saved to 'data/processed/customer_purchase_features.csv'")
    
    return customer_features

def train_purchase_model(customer_features, target_df):
    """Train a machine learning model to predict future purchases"""
    print("Training purchase prediction model...")
    
    # Merge features with target
    model_df = pd.merge(customer_features, target_df, on='CustomerID')
    
    # Select features and target
    X = model_df.drop(['CustomerID', 'Target'], axis=1)
    y = model_df['Target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Model evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top features by importance:")
    print(feature_importance.head())
    
    return model, scaler, feature_importance, X_test_scaled, y_test

def save_model_artifacts(model, scaler, feature_importance, output_dir='models'):
    """Save the model and related artifacts"""
    print("Saving model artifacts...")
    
    # Save model
    joblib.dump(model, f'{output_dir}/purchase_prediction_model.pkl')
    
    # Save scaler
    joblib.dump(scaler, f'{output_dir}/feature_scaler.pkl')
    
    # Save feature importance
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    print(f"Model artifacts saved to '{output_dir}' directory")

def create_predictions(df, model, scaler):
    """Create purchase predictions for all customers"""
    print("Creating purchase predictions for all customers...")
    
    # Create customer features
    customer_features = create_purchase_prediction_features(df)
    
    # Get CustomerIDs
    customer_ids = customer_features['CustomerID']
    
    # Remove CustomerID for prediction
    X = customer_features.drop('CustomerID', axis=1)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_prob = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'CustomerID': customer_ids,
        'PurchaseProbability': y_prob
    })
    
    # Sort by probability (descending)
    predictions_df = predictions_df.sort_values('PurchaseProbability', ascending=False)
    
    # Save predictions
    predictions_df.to_csv('data/processed/purchase_predictions.csv', index=False)
    print(f"Purchase predictions saved to 'data/processed/purchase_predictions.csv'")
    
    return predictions_df

def visualize_model_performance(model, X_test, y_test):
    """Create visualizations for model performance"""
    print("Creating model performance visualizations...")
    
    # ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Purchase Prediction Model')
    plt.legend()
    plt.savefig('visualizations/purchase_model_roc.png')
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Purchase Prediction Model')
    plt.savefig('visualizations/purchase_model_confusion_matrix.png')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.shape[1],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Features by Importance')
    plt.tight_layout()
    plt.savefig('visualizations/purchase_model_feature_importance.png')
    
    print("Model performance visualizations saved to 'visualizations/' directory")

def main():
    from data_processing import load_data, clean_retail_data
    
    # Load and clean data
    df = clean_retail_data(load_data())
    
    # Create target variable
    train_df, target_df = create_target_variable(df)
    
    # Create features
    customer_features = create_purchase_prediction_features(train_df)
    
    # Train model
    model, scaler, feature_importance, X_test, y_test = train_purchase_model(customer_features, target_df)
    
    # Save model artifacts
    save_model_artifacts(model, scaler, feature_importance)
    
    # Create predictions
    predictions_df = create_predictions(df, model, scaler)
    
    # Visualize model performance
    visualize_model_performance(model, X_test, y_test)
    
    return model, predictions_df

if __name__ == "__main__":
    main()