# Loan Approval Prediction using Machine Learning

A beginner-friendly machine learning project that analyzes loan approval data using multiple classification algorithms.

## Overview

This project demonstrates how to use machine learning to predict loan approval decisions based on various features in financial datasets. It implements a comprehensive pipeline including:

- Data loading and exploration
- Data visualization
- Preprocessing (handling missing values, encoding categorical features)
- Training and evaluating 11 different machine learning models
- Performance comparison across multiple datasets

## Features

- **Multi-format support**: Handles CSV, Excel (.xlsx, .xls), and Pickle (.pkl) files
- **Organized visualization folders**: Each dataset gets its own folder with descriptive filenames
- **Comprehensive visualizations**: Target distribution, feature correlations, boxplots, and more
- **Robust error handling**: Handles missing values, file loading issues, and optional dependencies
- **Performance comparisons**: Both visual and CSV exports of model performance metrics
- **Tabular results**: Clean, formatted tables with dividing lines for easy comparison

## Current Datasets

The project currently uses four real-world datasets:

1. **Original Loan Dataset** - Contains loan application data and approval status
2. **Financial Balance Sheet** - Financial records with credit decisioning information
3. **German Credit Risk** - The standard German credit dataset with risk classification
4. **Credit Risk Dataset** - Additional loan approval data with default status

## Requirements

To run this project, you need Python 3.6+ with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl (for Excel files)
- xlrd (for older Excel files)
- xgboost (optional, for XGBoost model)

Install the required packages using:
```
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl xlrd
```

For XGBoost support:
```
pip install xgboost
```

Note for macOS users: If you encounter XGBoost errors, install OpenMP:
```
brew install libomp
```

## How to Use

1. Place your loan/credit datasets in the appropriate folder
2. Update the dataset file paths in the `dataset_paths` dictionary at the bottom of the script:
   ```python
   dataset_paths = {
       "Original Loan Dataset": "/path/to/your/loan_approval_data.csv",
       "Financial Balance Sheet": "/path/to/your/balance_sheet.xlsx",
       "German Credit Risk": "/path/to/your/german_credit_data.csv",
       "Credit Risk Dataset": "/path/to/your/loan_approval_dataset_2.csv"
   }
   ```
3. Run the script:
   ```
   python loan_approval_prediction.py
   ```

## Supported File Formats

The script can handle multiple file formats:
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Pickle files (.pkl)

## Dataset Requirements

For each dataset:
- It should have at least 1500 records (you'll get a warning if fewer)
- The target variable (loan approval status) should be in the last column
- Should contain relevant features for loan/credit approval prediction

## Output Structure

The script organizes outputs in a clear folder structure:
```
├── Dataset_Name_visualizations/
│   ├── target_distribution.png
│   ├── categorical_feature_distribution.png
│   ├── correlation_heatmap.png
│   ├── boxplots.png
│   ├── pairplot.png
│   └── model_results/
│       ├── accuracy_comparison.png
│       └── model_confusion_matrix.png
└── comparison_results/
    ├── model_comparison.csv
    ├── performance_heatmap.png
    └── all_models_comparison.png
```

## Models Used

The project includes 11 machine learning models:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Decision Tree
6. Naive Bayes
7. Gradient Boosting
8. Genetic Algorithm (custom implementation)
9. XGBoost (if available)
10. AdaBoost
11. Neural Network (MLP)

## Model Comparison

The script generates a neatly formatted table comparing model performance across all datasets:

```
---------------------------------------------------------------------------------------------------
| Model                     | Original Loan Dataset | Financial Balance Sheet | Credit Risk Dataset | Average |
---------------------------------------------------------------------------------------------------
| XGBoost                   |               97.89% |                 100.0% |              97.89% |  98.59% |
---------------------------------------------------------------------------------------------------
| Gradient Boosting         |               97.54% |                 100.0% |              97.66% |   98.4% |
---------------------------------------------------------------------------------------------------
```

This allows for easy identification of the best performing models for your loan approval prediction tasks.

## Educational Focus

This project is designed to be educational and beginner-friendly with:
- Clear comments explaining each step
- Friendly output messages
- Visualizations to help understand the data
- Performance metrics to compare different algorithms
- Error handling with helpful messages
