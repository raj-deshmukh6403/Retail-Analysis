# 🛍️ Retail Customer Analysis Project

A comprehensive data analytics solution for retail businesses that includes customer segmentation, market basket analysis, purchase prediction, and a product recommendation system.

---

## 📊 Overview

This project analyzes retail transaction data to provide actionable insights for businesses, helping them understand customer behavior, optimize marketing strategies, and increase sales. The analysis pipeline includes:

- Data cleaning and preparation  
- Customer segmentation using RFM analysis  
- Market basket analysis through association rules mining  
- Purchase prediction modeling  
- Product recommendation system

---

## 🗂️ Project Structure

```
retail-analysis-project/
├── app.py                      # Web application entry point
├── main.py                     # Command-line pipeline script
├── data_processing.py          # Data cleaning and preparation
├── customer_segmentation.py    # RFM analysis
├── market_basket_analysis.py   # Association rules mining
├── purchase_prediction.py      # Purchase prediction model
├── recommendation_system.py    # Product recommendation system
├── templates/                  # Web app HTML templates
├── data/                       # Data directories
│   ├── raw/                    # Input data goes here
│   └── processed/              # Output data is saved here
├── models/                     # Saved models
├── visualizations/             # Generated charts
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Prerequisites

- Python 3.7+  
- pip (Python package manager)  
- Git (for cloning the repository)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/retail-analysis-project.git
cd retail-analysis-project
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate it on Windows
venv\Scripts\activate

# Activate it on macOS/Linux
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend flask openpyxl joblib apyori
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the **Online Retail dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail) and place it in the `data/raw/` directory.

But it is alredy there when cloning the repository, so no need to download

### 5. Create required directories

```bash
mkdir -p data/raw data/processed models visualizations
```
But the folders are alredy there when cloning the repository, so no need to craete them again
---

## 🧪 Usage

### ✅ Option 1: Command-Line Pipeline

Run the complete analysis pipeline:

```bash
python main.py
```

All results will be saved to their respective folders.

### 🌐 Option 2: Web Application

Start the web application:

```bash
python app.py
```

Then open your browser and go to:  
`http://127.0.0.1:5000`

You can:
- Upload retail data
- Run analysis interactively
- View/download results

---

## 📁 Output Files

Located in `data/processed/`:

- `cleaned_retail_data.csv`: Cleaned data  
- `customer_segments.csv`: RFM segmentation  
- `association_rules.csv`: Association rules  
- `customer_purchase_features.csv`: Features for prediction  
- `purchase_predictions.csv`: Predicted purchase likelihood  
- `product_recommendations.csv`: Personalized recommendations  

Visualizations will be in the `visualizations/` folder.

---

## 💼 Business Applications

- **Improve Marketing Effectiveness**: Target specific segments  
- **Optimize Product Placement**: Based on item affinities  
- **Increase Customer Retention**: Identify churn risks  
- **Boost Sales**: Recommend relevant products  
- **Enhance Inventory Management**: Forecast demand

---

## 🛠️ Troubleshooting

- **Memory Issues**: Use a subset of data  
- **Missing Columns**: Ensure your dataset includes all required fields  
- **Import Errors**: Recheck installed packages

---

## 🔮 Next Steps

- Import the output into **Power BI** or other tools  
- Build dashboards for various departments  
- Use segmentation insights for targeted outreach  
- Deploy the recommender system in production

---

## 🙌 Acknowledgements

- Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)  
- Created for educational purposes
