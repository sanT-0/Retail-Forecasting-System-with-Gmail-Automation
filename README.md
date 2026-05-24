# 🛒 Future Retail Sales Forecasting

> **An End-to-End Machine Learning Suite for Retail Sales Prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A comprehensive machine learning platform designed to predict future retail sales across multiple product categories with high accuracy. Features automated workflows, ensemble modeling, and email-based reporting.

---

## 📋 Table of Contents

- [Features](#-Features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Models](#-models)
- [Automation](#-automation)
- [API](#-api)
- [Configuration](#-configuration)
- [Contributing](#-contributing)

---

## ✨ Features

- **Multiple ML Models**: CatBoost, XGBoost, Linear Regression, Ridge, Gradient Boosting, and more
- **Ensemble Methods**: Weighted ensemble predictions for improved accuracy
- **Automated Workflows**: n8n integration for scheduled forecasting and email delivery
- **Multi-Category Support**: 10+ retail product categories pre-configured
- **Advanced Reporting**: HTML reports with visualizations and metrics
- **RESTful API**: Docker-ready forecast API for production deployment
- **Data Visualization**: Interactive charts and trend analysis
- **Feature Engineering**: Automated feature extraction and optimization

---

## 📁 Project Structure

```
FutureRetailSalesForecasting/
├── app.py                    # Main Streamlit/Flask application
├── main.py                   # Entry point for forecasting
├── diagnose.py               # Diagnostic and debugging utilities
├── requirements.txt          # Python dependencies
├── datasets/                 # Training datasets
│   ├── automotive_parts.csv
│   ├── beauty_cosmetics.csv
│   ├── electronics_gadgets.csv
│   └── generate_datasets.py  # Dataset generation script
├── models/                   # ML models and forecasters
│   ├── forecasters.py        # Core forecasting models
│   └── saved_models/         # Pre-trained model artifacts
├── utils/                    # Utility modules
│   ├── helpers.py            # Helper functions
│   ├── insights.py           # Data insights and analysis
│   └── visualise.py          # Visualization utilities
├── reports/                  # Report generation
│   ├── report_gen.py         # HTML report generator
│   └── formal_report/        # Generated reports
├── outputs/                  # Forecast outputs and results
├── n8n_automation/           # n8n workflow automation
│   ├── docker-compose.yml
│   ├── api/                  # REST API for forecasts
│   └── n8n_workflows/        # Workflow definitions
└── catboost_info/            # CatBoost training logs
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/FutureRetailSalesForecasting.git
cd FutureRetailSalesForecasting
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/Scripts/activate  # On Windows
source venv/bin/activate      # On macOS/Linux

# Or using conda
conda create -n sales-forecast python=3.9
conda activate sales-forecast
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### Run Basic Forecast

```bash
python main.py --category electronics_gadgets --model catboost
```

### Run Full Application

```bash
python app.py
```

### Generate Report

```bash
python reports/report_gen.py --category luxury_brand
```

---

## 💡 Usage

### Command Line Interface

```bash
# Forecast for a specific category
python main.py --category grocery_delivery --output forecast.csv

# Use specific model
python main.py --category toys_games --model ridge --alpha 1.0

# Ensemble prediction
python main.py --category fashion_apparel --ensemble

# Generate HTML report
python main.py --category electronics_gadgets --report
```

### Python API

```python
from models.forecasters import CatBoostForecaster
from datasets import load_dataset

# Load data
df = load_dataset('electronics_gadgets')

# Initialize and train forecaster
forecaster = CatBoostForecaster()
forecaster.fit(df)

# Make predictions
forecast = forecaster.predict(periods=30)
print(forecast)
```

### Data Insights

```python
from utils.insights import analyze_data
from utils.visualise import plot_forecast

df = load_dataset('luxury_brand')
insights = analyze_data(df)
plot_forecast(df, forecast)
```

---

## 📊 Datasets

Pre-configured datasets for various retail categories:

| Category | Filename | Records | Features |
|----------|----------|---------|----------|
| Electronics | electronics_gadgets.csv | ~500 | sales, trends, seasonality |
| Luxury | luxury_brand.csv | ~500 | premium products, trends |
| Grocery | grocery_delivery.csv | ~500 | daily sales, patterns |
| Fashion | fashion_apparel.csv | ~500 | seasonal data |
| Toys | toys_games.csv | ~500 | seasonal peaks |
| Automotive | automotive_parts.csv | ~500 | B2B patterns |
| Beauty | beauty_cosmetics.csv | ~500 | trend-driven |
| Furniture | furniture_decor.csv | ~500 | seasonal |
| Pharmacy | pharmacy_daily.csv | ~500 | daily patterns |
| Home Appliances | home_appliances.csv | ~500 | cyclical trends |

Generate new datasets:
```bash
python datasets/generate_datasets.py
```

---

## 🤖 Models

### Supported Models

- **CatBoost** - Gradient boosting with categorical feature support
- **XGBoost** - Extreme gradient boosting
- **Ridge Regression** - Regularized linear regression
- **Linear Regression** - Baseline linear model
- **HistGradientBoosting** - Histogram-based gradient boosting
- **ARIMA** - Statistical time series model
- **Weighted Ensemble** - Combination of multiple models

### Model Configuration

```python
from models.forecasters import (
    CatBoostForecaster, 
    XGBoostForecaster,
    EnsembleForecaster
)

# Single model
cat_model = CatBoostForecaster(iterations=100, depth=6)

# Ensemble
ensemble = EnsembleForecaster(
    models=['catboost', 'ridge', 'xgboost'],
    weights=[0.5, 0.3, 0.2]
)
```

---

## 🔄 Automation with n8n

The project includes n8n workflows for automated forecasting and reporting.

### Setup

```bash
cd n8n_automation
docker-compose up -d
```

### Features

- Scheduled forecasting (daily, weekly, monthly)
- Automated email delivery with reports
- Gmail integration for notifications
- API endpoints for real-time predictions

### Workflow Files

- `gmail_forecast_workflow.json` - Email-based reporting workflow

---

## 🌐 REST API

Docker-ready API for production deployment.

### Start API

```bash
cd n8n_automation/api
docker build -t forecast-api .
docker run -p 5000:5000 forecast-api
```

### API Endpoints

```bash
# Get forecast
POST /forecast
{
  "category": "electronics_gadgets",
  "periods": 30,
  "model": "catboost"
}

# Get metrics
GET /metrics/{category}

# Get available models
GET /models
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file:

```env
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
FORECAST_OUTPUT_PATH=./outputs
DEFAULT_MODEL=catboost
API_PORT=5000
```

### Model Parameters

Edit `models/forecasters.py` to customize:

```python
CATBOOST_PARAMS = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.05,
    'verbose': False
}
```

---

## 📈 Output Files

- `forecast_*.csv` - Prediction results
- `model_metrics.csv` - Performance metrics (MAE, RMSE, MAPE)
- `report.html` - Interactive HTML reports with charts
- `predictions.json` - JSON format predictions

---

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**Dataset Not Found**
```bash
python datasets/generate_datasets.py
```

**Model Training Fails**
```bash
python diagnose.py  # Run diagnostics
```

---

## 📦 Requirements

See [requirements.txt](requirements.txt) for full dependencies. Key packages:

- catboost
- xgboost
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- flask/streamlit
- docker

---

## 🔐 Security

- Keep `.env` file private (add to `.gitignore`)
- Use app-specific passwords for email
- Validate all API inputs
- Run API in secure Docker containers

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**sanT** - Created this comprehensive retail sales forecasting platform.
**Gmail** : santhoshsankar599@gmail.com
---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📞 Support

For issues, questions, or suggestions:
- Open an GitHub Issue
- Check existing documentation
- Run `diagnose.py` for system diagnostics
- **Gmail** : santhoshsankar599@gmail.com
---

## 🎯 Roadmap

- [ ] Web UI dashboard
- [ ] Real-time predictions
- [ ] Advanced anomaly detection
- [ ] GPU acceleration support
- [ ] Cloud deployment templates
- [ ] Mobile app integration

---

**Last Updated**: May 2026  
**Status**: Active Development ✅
