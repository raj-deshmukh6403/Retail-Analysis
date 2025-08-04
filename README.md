# 🛒 Retail Customer Analysis Project

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/ML-Customer%20Analytics-red.svg)](https://github.com/raj-deshmukh6403/Retail-Analysis)

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Modules](#analysis-modules)
- [Web Application](#web-application)
- [Output Files](#output-files)
- [Visualizations](#visualizations)
- [Frontend Screenshots](#frontend-screenshots)
- [Business Applications](#business-applications)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This comprehensive retail customer analysis project leverages machine learning and data analytics to extract actionable insights from retail transaction data. The system provides end-to-end analysis including customer segmentation, market basket analysis, purchase prediction, and product recommendations.

The project includes both a command-line interface for batch processing and a Flask web application for interactive analysis with real-time results and visualizations.

## ✨ Features

### 🔍 Core Analytics Features
- **Customer Segmentation**: RFM (Recency, Frequency, Monetary) analysis with 13 distinct customer segments
- **Market Basket Analysis**: Apriori algorithm for discovering product associations and cross-selling opportunities
- **Purchase Prediction**: Random Forest classifier to predict future customer purchase behavior
- **Recommendation System**: Collaborative filtering for personalized product recommendations
- **Data Processing**: Comprehensive data cleaning and feature engineering pipeline

### 🖥️ Interactive Features
- **Web Interface**: Flask-based web application for easy file upload and analysis
- **Real-time Processing**: Live analysis of uploaded retail data
- **Interactive Visualizations**: Charts and graphs for immediate insights
- **Export Capabilities**: Download all results in a single ZIP file
- **File Preview**: In-browser CSV file viewing and image visualization

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Flask**: Web framework for the interactive application

### Machine Learning & Analytics
- **Random Forest**: Purchase prediction modeling
- **K-Nearest Neighbors**: Collaborative filtering for recommendations
- **Apriori Algorithm**: Market basket analysis (via mlxtend)
- **RFM Analysis**: Customer segmentation methodology

### Visualization & UI
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization
- **Bootstrap 5**: Responsive web interface
- **HTML/CSS**: Frontend development

### Data Processing
- **Pandas**: Data cleaning and transformation
- **Joblib**: Model serialization
- **Openpyxl**: Excel file processing

## 📁 Project Structure

```
retail_analysis_project/
│
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 LICENSE                      # MIT License
│
├── 🗂️ assets/                      # Static assets and documentation
│   └── screenshots/                # Frontend screenshots
│       ├── homepage.png
│       ├── upload-interface.png
│       ├── analysis-progress.png
│       ├── results-dashboard.png
│       ├── data-preview.png
│       ├── visualizations-gallery.png
│       └── download-results.png
│
├── 🗂️ data/
│   ├── raw/                        # Raw data files
│   │   └── Online_Retail.xlsx     # Input dataset
│   └── processed/                  # Processed data outputs
│       ├── cleaned_retail_data.csv
│       ├── customer_segments.csv
│       ├── association_rules.csv
│       ├── purchase_predictions.csv
│       └── product_recommendations.csv
│
├── 🗂️ models/                      # Trained ML models
│   ├── purchase_prediction_model.pkl
│   ├── feature_scaler.pkl
│   └── feature_importance.csv
│
├── 🗂️ visualizations/              # Generated charts and graphs
│   ├── customer_segments.png
│   ├── top_associations.png
│   ├── purchase_model_roc.png
│   └── top_recommended_products.png
│
├── 🗂️ templates/                   # HTML templates for web app
│   ├── index.html
│   ├── results.html
│   └── view_csv.html
│
├── 🐍 Core Python Modules
├── 📄 main.py                      # Command-line interface
├── 📄 app.py                       # Flask web application
├── 📄 data_processing.py           # Data cleaning and preparation
├── 📄 customer_segmentation.py     # RFM analysis and segmentation
├── 📄 market_basket_analysis.py    # Association rules mining
├── 📄 purchase_prediction.py       # ML model for purchase prediction
└── 📄 recommendation_system.py     # Collaborative filtering recommendations
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/raj-deshmukh6403/Retail-Analysis.git
cd retail_analysis_project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create Required Directories
```bash
# The application will create these automatically, but you can create them manually:
mkdir data\raw data\processed models visualizations
```

## 📊 Usage

### Option 1: Command Line Interface

1. **Place your data file** in the `data/raw/` directory (should be named `Online_Retail.xlsx`)

2. **Run the complete analysis pipeline:**
```bash
python main.py
```

This will execute all analysis modules sequentially and generate outputs in the respective directories.

### Option 2: Web Application

1. **Start the Flask web server:**
```bash
python app.py
```

2. **Open your browser** and navigate to: `http://localhost:5000`

3. **Upload your data file** using the web interface (supports .xlsx and .csv files)

4. **View results** in real-time with interactive visualizations

5. **Download all results** as a ZIP file for further analysis

## 🔬 Analysis Modules

### 1. Data Processing (`data_processing.py`)
- **Data Loading**: Supports Excel and CSV formats
- **Data Cleaning**: Removes invalid transactions, handles missing values
- **Feature Engineering**: Creates derived features like TotalAmount
- **Data Validation**: Ensures data quality and consistency

**Key Functions:**
- `load_data()`: Import raw data files
- `clean_retail_data()`: Comprehensive data cleaning
- `save_processed_data()`: Export cleaned dataset

### 2. Customer Segmentation (`customer_segmentation.py`)
- **RFM Analysis**: Calculates Recency, Frequency, and Monetary metrics
- **Customer Scoring**: Creates quartile-based RFM scores
- **Segment Classification**: 13 distinct customer segments including:
  - 🏆 Champions: Best customers with high value and engagement
  - 💎 Loyal Customers: Regular, valuable customers
  - 🌟 Potential Loyalists: Recent customers with good potential
  - ⚠️ At Risk: Previously valuable customers showing decline
  - 😴 Hibernating: Customers who haven't purchased recently

**Key Functions:**
- `create_rfm_features()`: Generate RFM metrics and segments
- `save_customer_segments()`: Export segmentation results
- `visualize_segments()`: Create segment distribution charts

### 3. Market Basket Analysis (`market_basket_analysis.py`)
- **Association Rule Mining**: Uses Apriori algorithm
- **Product Relationships**: Identifies frequently bought together items
- **Cross-selling Opportunities**: Discovers product associations
- **Memory Optimization**: Handles large datasets efficiently

**Key Metrics:**
- **Support**: Frequency of itemset occurrence
- **Confidence**: Likelihood of consequent given antecedent
- **Lift**: Strength of association between items

**Key Functions:**
- `create_basket_analysis()`: Generate association rules
- `save_association_rules()`: Export discovered rules
- `visualize_top_associations()`: Create association visualizations

### 4. Purchase Prediction (`purchase_prediction.py`)
- **Target Variable Creation**: Predicts future purchases within 30 days
- **Feature Engineering**: Creates comprehensive customer features
- **Model Training**: Random Forest classifier with cross-validation
- **Performance Evaluation**: ROC curves, confusion matrix, feature importance

**Prediction Features:**
- Purchase frequency and recency
- Customer tenure and loyalty
- Product diversity and preferences
- Spending patterns and trends

**Key Functions:**
- `create_target_variable()`: Define prediction target
- `train_purchase_model()`: Train and evaluate ML model
- `create_predictions()`: Generate customer purchase probabilities
- `visualize_model_performance()`: Create performance charts

### 5. Recommendation System (`recommendation_system.py`)
- **Collaborative Filtering**: Customer-based recommendations
- **Similarity Calculation**: Uses cosine similarity metrics
- **Personalized Suggestions**: Tailored product recommendations
- **Scalable Architecture**: Handles large customer bases

**Key Functions:**
- `create_recommendation_system()`: Generate product recommendations
- `save_recommendations()`: Export recommendation lists
- `visualize_top_recommended_products()`: Create recommendation charts

## 🌐 Web Application

The Flask web application provides an intuitive interface for non-technical users:

### Features:
- **📤 File Upload**: Drag-and-drop or browse file selection
- **⚡ Real-time Processing**: Live analysis with progress indicators
- **📊 Interactive Results**: View data tables and visualizations
- **💾 Bulk Download**: Export all results in a single ZIP file
- **🖼️ Image Preview**: In-browser visualization viewing
- **📄 CSV Preview**: Table view for processed data files

### Routes:
- `/`: Home page with file upload interface
- `/upload`: Handle file upload and processing
- `/analyze/<filename>`: Run analysis pipeline
- `/view_file/<folder>/<filename>`: Preview generated files
- `/download_all`: Download complete results package

## 📈 Output Files

### Processed Data Files (CSV format):
1. **`cleaned_retail_data.csv`**: Processed transaction data
2. **`customer_segments.csv`**: RFM analysis results with segment classifications
3. **`association_rules.csv`**: Market basket analysis rules with metrics
4. **`customer_purchase_features.csv`**: Engineered features for ML modeling
5. **`purchase_predictions.csv`**: Customer purchase probability scores
6. **`product_recommendations.csv`**: Personalized product recommendations

### Model Artifacts:
1. **`purchase_prediction_model.pkl`**: Trained Random Forest model
2. **`feature_scaler.pkl`**: StandardScaler for feature normalization
3. **`feature_importance.csv`**: Feature importance scores from the model

### Visualizations (PNG format):
1. **`customer_segments.png`**: Customer segment distribution
2. **`top_associations.png`**: Top product associations by lift
3. **`support_confidence_scatter.png`**: Support vs confidence analysis
4. **`purchase_model_roc.png`**: ROC curve for purchase prediction
5. **`purchase_model_confusion_matrix.png`**: Model confusion matrix
6. **`top_recommended_products.png`**: Most recommended products
7. **`recommendation_distribution.png`**: Recommendation frequency distribution

## 🎨 Visualizations

The project generates comprehensive visualizations for business insights:

### Customer Analytics:
- **Segment Distribution**: Bar charts showing customer segment sizes
- **RFM Heatmaps**: Visual representation of customer value matrices
- **Customer Journey**: Timeline analysis of customer behavior

### Product Analytics:
- **Association Networks**: Visual representation of product relationships
- **Recommendation Charts**: Top recommended products and frequencies
- **Cross-selling Opportunities**: Product pairing visualizations

### Model Performance:
- **ROC Curves**: Model performance evaluation
- **Feature Importance**: Most influential prediction factors
- **Confusion Matrices**: Classification accuracy assessment

## 📸 Frontend Screenshots

### Web Application Interface

<div align="center">

#### 🏠 Homepage & Upload Interface
![Homepage](https://raw.githubusercontent.com/raj-deshmukh6403/Retail-Analysis/main/assets/screenshots/homepage.png)
*Clean, intuitive homepage with file upload interface*

#### 📤 File Upload & Processing
![Upload Interface](https://raw.githubusercontent.com/raj-deshmukh6403/Retail-Analysis/main/assets/screenshots/upload-interface.png)
*Drag-and-drop file upload with real-time validation*

#### ⚡ Analysis in Progress
![Analysis Progress](https://raw.githubusercontent.com/raj-deshmukh6403/Retail-Analysis/main/assets/screenshots/analysis-progress.png)
*Live processing with progress indicators*

#### 📊 Results Dashboard
![Results Dashboard](https://raw.githubusercontent.com/raj-deshmukh6403/Retail-Analysis/main/assets/screenshots/results-dashboard.png)
*Comprehensive results page with interactive visualizations*

#### 📄 Data Preview
![Data Preview](https://raw.githubusercontent.com/raj-deshmukh6403/Retail-Analysis/main/assets/screenshots/data-preview.png)
*In-browser CSV file viewing with formatted tables*

#### 🎨 Visualizations Gallery
![Visualizations Gallery](https://raw.githubusercontent.com/raj-deshmukh6403/Retail-Analysis/main/assets/screenshots/visualizations-gallery.png)
*Interactive chart gallery with thumbnail previews*

#### 💾 Download Results
![Download Results](https://raw.githubusercontent.com/raj-deshmukh6403/Retail-Analysis/main/assets/screenshots/download-results.png)
*One-click download of all analysis results*

</div>

## 🏢 Business Applications

### Marketing & CRM:
- **Targeted Campaigns**: Use customer segments for personalized marketing
- **Customer Retention**: Identify at-risk customers for retention campaigns
- **Cross-selling**: Implement association rules for product recommendations
- **Customer Lifetime Value**: Predict and optimize customer value

### E-commerce & Retail:
- **Product Placement**: Optimize store layouts based on associations
- **Inventory Management**: Stock planning based on purchase predictions
- **Recommendation Engines**: Implement collaborative filtering
- **Promotional Strategies**: Design offers based on customer segments

### Business Intelligence:
- **Dashboard Creation**: Import CSV files into Power BI, Tableau, or Excel
- **KPI Monitoring**: Track segment performance and trends
- **Predictive Analytics**: Use models for forecasting and planning
- **ROI Analysis**: Measure campaign effectiveness by segment

### Strategic Planning:
- **Market Analysis**: Understand customer behavior patterns
- **Product Development**: Identify popular product combinations
- **Customer Acquisition**: Target similar profiles to best customers
- **Revenue Optimization**: Focus on high-value customer segments

## 🔧 Configuration & Customization

### Model Parameters:
```python
# Adjust in respective modules
MIN_SUPPORT = 0.01          # Market basket analysis threshold
MIN_CONFIDENCE = 0.2        # Association rule confidence
FUTURE_DAYS = 30           # Purchase prediction window
N_RECOMMENDATIONS = 5       # Number of recommendations per customer
```

### Data Requirements:
The system expects retail transaction data with these columns:
- `InvoiceNo`: Transaction identifier
- `StockCode`: Product identifier
- `Description`: Product description
- `Quantity`: Items purchased
- `InvoiceDate`: Transaction timestamp
- `UnitPrice`: Price per unit
- `CustomerID`: Customer identifier
- `Country`: Customer location (optional)

## 📊 Performance Considerations

### Memory Optimization:
- Market basket analysis limited to top 1000 products
- Sparse matrix representations for efficiency
- Incremental processing for large datasets

### Scalability:
- Modular design allows for distributed processing
- Database integration possible for larger datasets
- Caching mechanisms for improved performance

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Guidelines:
- Follow PEP 8 coding standards
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Raj Deshmukh**
- GitHub: [@raj-deshmukh6403](https://github.com/raj-deshmukh6403)
- Project Link: [https://github.com/raj-deshmukh6403/Retail-Analysis](https://github.com/raj-deshmukh6403/Retail-Analysis)

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Online Retail dataset
- **Scikit-learn** community for excellent ML libraries
- **Flask** framework for web application capabilities
- **Bootstrap** for responsive UI components
- **Open source community** for various supporting libraries

---

<div align="center">

### 🌟 If you find this project useful, please give it a star! ⭐

**Made with ❤️ for the retail analytics community**

</div>
