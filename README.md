# 💳 Credit Class Prediction using Machine Learning

📌 Project Overview
This project focuses on predicting credit class eligibility using Machine Learning techniques.
The objective is to analyze customer credit-related data, perform preprocessing and feature engineering, handle class imbalance, and apply a supervised learning model to predict credit classes accurately.

The project follows a modular pipeline-based approach, making it scalable, reusable, and suitable for real-world ML applications.

# 🧠 Prerequisites

Before working on this project, you should have a basic understanding of:

- Python programming
- Machine Learning fundamentals
- Supervised learning algorithms
- Data preprocessing and feature engineering concepts

# 🛠️ Technologies & Libraries Used

This project is implemented entirely in Python using the following libraries:

NumPy – Numerical computations

Pandas – Data manipulation and analysis

Matplotlib & Seaborn – Data visualization

Scikit-learn – Machine learning algorithms and preprocessing

Imbalanced-learn – Handling imbalanced datasets

Flask – Model deployment (API)

Logging – Tracking pipeline execution and errors

# ⚙️ Installation Guide

## Step 1: Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate     # macOS

## Step 2: Install required dependencies
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install imbalanced-learn
pip install flask

## Step 3: Save dependencies
pip freeze > requirements.txt

# ✅ Verifying Installation

You can verify successful installation by importing the required modules:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os
import warnings

warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from random_sample import random_sample_imputation_technique
from var_out import variable_trainsormation_outliers
from feature_selection import complete_feature_selection
from imbalance_data import balance_data

# 🧩 Project Structure

creditcard/
│
├── data/
│   └── credit_data.csv
│
├── log_files/
│   └── main.log
│
├── log_code.py
├── random_sample.py
├── var_out.py
├── feature_selection.py
├── imbalance_data.py
├── main.py
├── requirements.txt
└── README.md

# 🔍 Dataset Information

Title: Credit Card Prediction Dataset

Source: Kaggle
Description:
The dataset contains customer financial and demographic attributes used to predict credit class eligibility.

# 📊 Methodology & Approach
1. Data Loading & Validation
Data is loaded using Pandas
Logging is implemented to track pipeline execution

2. Missing Value Treatment
Random sample imputation technique is applied

3. Outlier Detection & Variable Transformation
Outliers are identified and treated
Feature transformations are applied where required

4. Feature Selection
Relevant features are selected to improve model performance and reduce overfitting

5. Handling Imbalanced Data
Class imbalance is addressed using resampling techniques from imbalanced-learn

6. Model Selection
Linear Regression was selected based on ROC Curve analysis
The model was evaluated against other algorithms and chosen for optimal performance

# 🚀 Model Deployment
The trained model is deployed using Flask, providing a REST API for predictions.

🔗 Link:
[(https://isha22.pythonanywhere.com)]

📈 Evaluation Metrics
- ROC Curve
- Accuracy
- Precision
- Recall

# 🎯 Key Highlights

- Modular and reusable ML pipeline
- Robust logging mechanism
- Handles missing values, outliers, and imbalanced data
- End-to-end ML lifecycle: data → model → deployment

# 📌 Future Enhancements

- Try advanced models (Random Forest, XGBoost)
- Hyperparameter tuning
- Model monitoring and retraining
- Front-end integration for user interaction

# Author

Isha, 
MBA (Business Analytics),
Machine Learning | Data Analytics
