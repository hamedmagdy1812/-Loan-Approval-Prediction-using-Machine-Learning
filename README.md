# Loan Approval Prediction using Machine Learning

A beginner-friendly machine learning project that analyzes loan approval data using multiple classification algorithms.

## Overview

This project demonstrates how to use machine learning to predict loan approval decisions based on various features in financial datasets. It implements a comprehensive pipeline including:

- Data loading and exploration
- Data visualization
- Preprocessing (handling missing values, encoding categorical features)
- Training and evaluating 8 different machine learning models
- Performance comparison across multiple datasets

## Requirements

To run this project, you need Python 3.6+ with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl (for Excel files)
- xlrd (for older Excel files)

Install the required packages using:
```
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl xlrd
```

## How to Use

1. Place your loan/credit datasets in the appropriate folder
2. Update the dataset file paths in the `dataset_paths` dictionary at the bottom of the script:
   ```python
   dataset_paths = {
       "Credit Score Dataset": "/path/to/your/credit_data.csv",
       "Loan Approval Dataset": "/path/to/your/loan_data.xlsx",
       "Financial History Dataset": "/path/to/your/financial_data.pkl"
   }
   ```
3. Run the script:
   ```
   python loan_approval_prediction.py
   ```

## Supported File Formats

The script can now handle multiple file formats:
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Pickle files (.pkl)

## Dataset Requirements

For each dataset:
- It should have at least 1500 records (you'll get a warning if fewer)
- The target variable (loan approval status) should be in the last column
- Should contain relevant features for loan/credit approval prediction

## Output

The script will:
1. Display information about each dataset
2. Create visualizations saved as PNG files
3. Train and evaluate 8 machine learning models on each dataset
4. Generate a performance comparison chart across all datasets
5. Save the comparison as 'model_comparison.png'

## Error Handling

The script now includes improved error handling:
- Catches and reports file loading errors
- Provides warnings for datasets with fewer than 1500 records
- Gracefully handles visualization errors
- Reports preprocessing and model training errors

## Models Used

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Decision Tree
6. Naive Bayes
7. Gradient Boosting
8. Genetic Algorithm (custom implementation)

## Educational Focus

This project is designed to be educational and beginner-friendly with:
- Clear comments explaining each step
- Friendly output messages
- Visualizations to help understand the data
- Performance metrics to compare different algorithms
