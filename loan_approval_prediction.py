#!/usr/bin/env python3
# Loan Approval Prediction using Machine Learning
# A beginner-friendly educational project

# Let's import all the libraries we'll need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
import warnings
warnings.filterwarnings('ignore')

# Setting up visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.ioff()  # Turn off interactive mode to prevent script from waiting for plot windows to close

# Simple Genetic Algorithm Classifier
class GeneticAlgorithmClassifier:
    """A simple genetic algorithm for classification problems"""
    def __init__(self, pop_size=100, generations=50, mutation_rate=0.1):
        self.pop_size = pop_size
        self.generations = generations 
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.feature_weights = None
        
    def _fitness(self, solution, X, y):
        """Calculate fitness using weighted features and simple threshold"""
        predictions = (X * solution).sum(axis=1) > 0.5
        return accuracy_score(y, predictions)
    
    def _create_population(self, n_features):
        """Create initial population with random weights"""
        return [np.random.uniform(-1, 1, n_features) for _ in range(self.pop_size)]
    
    def _select_parents(self, population, fitness_scores):
        """Tournament selection for parents"""
        parents = []
        for _ in range(len(population)):
            idx1, idx2 = random.sample(range(len(population)), 2)
            if fitness_scores[idx1] > fitness_scores[idx2]:
                parents.append(population[idx1])
            else:
                parents.append(population[idx2])
        return parents
    
    def _crossover(self, parents):
        """Create new generation through crossover"""
        children = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):
                crossover_point = random.randint(1, len(parents[i])-1)
                child1 = np.concatenate([parents[i][:crossover_point], parents[i+1][crossover_point:]])
                child2 = np.concatenate([parents[i+1][:crossover_point], parents[i][crossover_point:]])
                children.extend([child1, child2])
        return children
    
    def _mutate(self, population):
        """Apply random mutations to population"""
        for i in range(len(population)):
            for j in range(len(population[i])):
                if random.random() < self.mutation_rate:
                    population[i][j] += random.uniform(-0.5, 0.5)
        return population
    
    def fit(self, X, y):
        """Train the genetic algorithm classifier"""
        X = np.array(X)
        n_features = X.shape[1]
        population = self._create_population(n_features)
        
        for _ in range(self.generations):
            fitness_scores = [self._fitness(solution, X, y) for solution in population]
            best_idx = np.argmax(fitness_scores)
            self.best_solution = population[best_idx]
            
            parents = self._select_parents(population, fitness_scores)
            children = self._crossover(parents)
            population = self._mutate(children)
        
        self.feature_weights = self.best_solution
        return self
    
    def predict(self, X):
        """Make predictions with trained model"""
        X = np.array(X)
        return (X * self.feature_weights).sum(axis=1) > 0.5

# Function to load dataset from different file formats
def load_dataset(file_path):
    """Load dataset from various file formats (CSV, Excel, Pickle)"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif file_ext == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {str(e)}")
        return None

# Function to visualize the dataset
def visualize_data(df, dataset_name):
    """Create visualizations to better understand the dataset"""
    print("\n📊 Let's visualize the data to better understand it!")
    
    plt.figure(figsize=(18, 12))
    
    # Target variable distribution
    plt.subplot(2, 2, 1)
    target_col = df.columns[-1]  # Assuming target is last column
    if df[target_col].dtype == 'object':
        sns.countplot(x=target_col, data=df)
        plt.title(f"Distribution of {target_col}")
    else:
        sns.histplot(df[target_col], kde=True)
        plt.title(f"Distribution of {target_col}")
    
    # Numerical features distributions
    plt.subplot(2, 2, 2)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns[:3]
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            sns.histplot(df[col], kde=True, label=col)
        plt.title("Distribution of Key Numerical Features")
        plt.legend()
    
    # Correlation heatmap
    plt.subplot(2, 2, 3)
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = numerical_df.corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, mask=mask, cmap="coolwarm", linewidths=0.5, fmt=".2f")
    plt.title("Correlation Heatmap")
    
    # Box plots
    plt.subplot(2, 2, 4)
    if len(numerical_cols) > 0:
        sns.boxplot(data=df[numerical_cols])
        plt.title("Box Plots of Numerical Features")
    
    plt.tight_layout()
    # Save the figure but don't display it
    filename = f"{dataset_name}_visualizations.png"
    plt.savefig(filename)
    plt.close()  # Close the figure to release memory
    print(f"✅ Visualizations saved to '{filename}'")

# Function to preprocess the dataset
def preprocess_data(df):
    """Preprocess the dataset: handle missing values, encode categorical features, scale numerical data"""
    print("\n🧹 Time to clean and prepare our data!")
    
    print("\n📋 Data Types and Missing Values Summary:")
    print(df.dtypes)
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    
    # Identify target column (assuming it's the last column)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"\n🔢 We have {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features.")
    
    # Create preprocessor pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"🔍 Training set: {X_train.shape[0]} samples")
    print(f"🔍 Testing set: {X_test.shape[0]} samples")
    
    # Encode target variable if categorical
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
    
    return X_train, X_test, y_train, y_test, preprocessor

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate multiple machine learning models"""
    print("\n🤖 Let's train and evaluate our machine learning models!")
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Genetic Algorithm": GeneticAlgorithmClassifier(generations=20)
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Training {name}...")
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        # Print results
        print(f"✅ {name} trained successfully!")
        print(f"📈 Accuracy: {accuracy * 100:.2f}%")
        print("\n📊 Confusion Matrix:")
        print(conf_matrix)
        print("\n📋 Classification Report:")
        print(class_report)
    
    return results

# Function to process each dataset
def process_dataset(file_path, dataset_name):
    """Process a single dataset through the entire machine learning pipeline"""
    print(f"\n{'='*80}")
    print(f"✨ DATASET: {dataset_name} ✨".center(80))
    print(f"{'='*80}\n")
    
    # Step 1: Load the data
    print(f"🚀 Let's start exploring the {dataset_name} dataset!")
    df = load_dataset(file_path)
    
    if df is None:
        print(f"❌ Could not process {dataset_name}. Skipping to the next dataset.")
        return None
        
    # Check if dataset has enough records
    if len(df) < 1500:
        print(f"⚠️ Warning: This dataset has only {len(df)} records, which is less than the recommended 1500 records.")
        proceed = input("Do you want to continue with this dataset anyway? (y/n): ")
        if proceed.lower() != 'y':
            return None
    
    print("👀 Here's a peek at the first few rows:")
    print(df.head())
    print(f"\n📊 This dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Step 2: Visualize the data
    try:
        visualize_data(df, dataset_name)
    except Exception as e:
        print(f"⚠️ Warning: Could not create visualizations: {str(e)}")
    
    # Step 3: Preprocess the data
    try:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return None
    
    # Step 4: Train and evaluate models
    try:
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
        return results
    except Exception as e:
        print(f"❌ Error during model training/evaluation: {str(e)}")
        return None

# Function to compare model performances across datasets
def compare_models(all_results):
    """Create a comparison table of model performances across all datasets"""
    print(f"\n{'='*80}")
    print("📊 MODEL PERFORMANCE COMPARISON 📊".center(80))
    print(f"{'='*80}\n")
    
    # Create comparison DataFrame
    comparison = pd.DataFrame()
    
    # Get accuracies for each dataset and model (as percentages)
    for dataset, results in all_results.items():
        accuracies = {model: result['accuracy'] * 100 for model, result in results.items()}
        comparison[dataset] = pd.Series(accuracies)
    
    # Add average performance column
    comparison['Average'] = comparison.mean(axis=1)
    
    # Sort by average performance
    comparison = comparison.sort_values('Average', ascending=False)
    
    # Format percentages
    comparison = comparison.round(2)
    comparison_display = comparison.copy()
    for col in comparison.columns:
        comparison_display[col] = comparison_display[col].astype(str) + '%'
    
    print(comparison_display)
    print("\n📌 Best Overall Model: " + comparison.index[0])
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    comparison.plot(kind='bar')
    plt.title('Model Performance Comparison Across Datasets (%)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()  # Close the figure
    
    print("✅ Comparison chart saved to 'model_comparison.png'")

# Main execution flow
if __name__ == "__main__":
    print("\n" + "✨ LOAN APPROVAL PREDICTION PROJECT ✨".center(80))
    print("A beginner-friendly machine learning tutorial".center(80) + "\n")
    
    # Define paths to your local datasets - update these with your actual file paths
    dataset_paths = {
        "Loan Approval Dataset": "/Users/Hamed/Documents/selected topics/-Loan-Approval-Prediction-using-Machine-Learning/Datasets/loan_approval_data.csv",
        "Financial History Dataset": "/Users/Hamed/Documents/selected topics/-Loan-Approval-Prediction-using-Machine-Learning/Datasets/Balance Sheet .xlsx"
    }
    
    # Process each dataset and collect results
    all_results = {}
    
    for name, path in dataset_paths.items():
        try:
            print(f"\n🔍 Processing the {name} dataset from {path}...")
            results = process_dataset(path, name)
            if results:
                all_results[name] = results
        except Exception as e:
            print(f"❌ Error processing {name}: {str(e)}")
    
    # Compare model performances if we have results from multiple datasets
    if len(all_results) > 1:
        try:
            compare_models(all_results)
        except Exception as e:
            print(f"❌ Error creating comparison: {str(e)}")
    elif len(all_results) == 1:
        print("\n✅ Successfully processed one dataset. Need at least two datasets for comparison.")
    else:
        print("\n❌ No datasets were successfully processed. Please check your file paths and formats.")
    
    print("\n🎉 That's all, folks! Thanks for exploring loan approval prediction with us! 🎉") 