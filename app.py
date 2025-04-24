# app.py
import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from flask import send_file
from werkzeug.utils import secure_filename
import pickle
import zipfile
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data, clean_retail_data, save_processed_data
from customer_segmentation import create_rfm_features, save_customer_segments, visualize_segments
from market_basket_analysis import create_basket_analysis, save_association_rules, visualize_top_associations
from purchase_prediction import create_target_variable, create_purchase_prediction_features, train_purchase_model, save_model_artifacts, create_predictions, visualize_model_performance
from recommendation_system import create_recommendation_system, save_recommendations, visualize_top_recommended_products

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.secret_key = "retail_analysis_secret_key"
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'data/raw')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'data/processed')
app.config['VISUALIZATION_FOLDER'] = os.path.join(BASE_DIR, 'visualizations')
app.config['MODEL_FOLDER'] = os.path.join(BASE_DIR, 'models')

# Ensure required directories exist
for directory in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], 
                app.config['VISUALIZATION_FOLDER'], app.config['MODEL_FOLDER']]:
    os.makedirs(directory, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash(f'File {filename} uploaded successfully')
        
        return redirect(url_for('analyze', filename=filename))
    else:
        flash('Invalid file type. Please upload an Excel or CSV file.')
        return redirect(request.url)

@app.route('/analyze/<filename>')
def analyze(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Run the analysis pipeline
        df = load_data(file_path)
        df_clean = clean_retail_data(df)
        save_processed_data(df_clean)
        
        # Run all analysis steps
        # Customer Segmentation
        rfm_df = create_rfm_features(df_clean)
        save_customer_segments(rfm_df)
        visualize_segments(rfm_df)
        
        # Market Basket Analysis
        association_rules_df, basket_sets = create_basket_analysis(df_clean)
        save_association_rules(association_rules_df)
        visualize_top_associations(association_rules_df)
        
        # Purchase Prediction
        train_df, target_df = create_target_variable(df_clean)
        customer_features = create_purchase_prediction_features(train_df)
        model, scaler, feature_importance, X_test, y_test = train_purchase_model(customer_features, target_df)
        save_model_artifacts(model, scaler, feature_importance)
        prediction_df = create_predictions(df_clean, model, scaler)
        visualize_model_performance(model, X_test, y_test)
        
        # Recommendation System
        recommendations_df = create_recommendation_system(df_clean)
        save_recommendations(recommendations_df)
        visualize_top_recommended_products(recommendations_df)
        
        # Get the list of generated files
        processed_files = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if os.path.isfile(os.path.join(app.config['PROCESSED_FOLDER'], f))]
        visualization_files = [f for f in os.listdir(app.config['VISUALIZATION_FOLDER']) if os.path.isfile(os.path.join(app.config['VISUALIZATION_FOLDER'], f))]
        
        return render_template('results.html', 
                              processed_files=processed_files,
                              visualization_files=visualization_files)
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download_all')
def download_all():
    # Create in-memory zip file
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        # Add processed data files
        for folder, subfolders, files in os.walk(app.config['PROCESSED_FOLDER']):
            for file in files:
                file_path = os.path.join(folder, file)
                arcname = os.path.join('processed', file)
                zf.write(file_path, arcname)
        
        # Add visualization files
        for folder, subfolders, files in os.walk(app.config['VISUALIZATION_FOLDER']):
            for file in files:
                file_path = os.path.join(folder, file)
                arcname = os.path.join('visualizations', file)
                zf.write(file_path, arcname)
                
        # Add model files
        for folder, subfolders, files in os.walk(app.config['MODEL_FOLDER']):
            for file in files:
                file_path = os.path.join(folder, file)
                arcname = os.path.join('models', file)
                zf.write(file_path, arcname)
    
    # Reset file pointer
    memory_file.seek(0)
    
    # Return the in-memory zip file
    return send_file(
        memory_file,
        download_name='retail_analysis_results.zip',
        as_attachment=True,
        mimetype='application/zip'
    )

@app.route('/view_file/<folder>/<filename>')
def view_file(folder, filename):
    # Determine the folder path based on the folder parameter
    if folder == 'processed':
        folder_path = app.config['PROCESSED_FOLDER']
    elif folder == 'visualizations':
        folder_path = app.config['VISUALIZATION_FOLDER']
    else:
        flash('Invalid folder')
        return redirect(url_for('index'))
    
    file_path = os.path.join(folder_path, filename)
    
    # For CSV files, display content
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            return render_template('view_csv.html', filename=filename, 
                                  table=df.head(100).to_html(classes='table table-striped'))
        except Exception as e:
            flash(f'Error reading file: {str(e)}')
            return redirect(url_for('results'))
    
    # For images, serve directly
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        return send_from_directory(folder_path, filename)
    
    # For other file types
    else:
        flash('File type not supported for preview')
        return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)