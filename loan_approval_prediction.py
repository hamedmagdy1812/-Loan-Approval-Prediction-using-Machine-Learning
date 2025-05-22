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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
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
    
    # Create a folder for this dataset's visualizations
    viz_folder = f"{dataset_name}_visualizations"
    os.makedirs(viz_folder, exist_ok=True)
    print(f"✅ Created folder '{viz_folder}' for visualizations")
    
    # 1. Target variable distribution
    plt.figure(figsize=(10, 7))
    target_col = df.columns[-1]  # Assuming target is last column
    if df[target_col].dtype == 'object':
        ax = sns.countplot(x=target_col, data=df)
        plt.title(f"Distribution of {target_col}")
        # Rotate x labels and adjust their position
        plt.xticks(rotation=45, ha='right')
        # Adjust bottom margin to make room for labels
        plt.subplots_adjust(bottom=0.15)
    else:
        sns.histplot(df[target_col], kde=True)
        plt.title(f"Distribution of {target_col}")
    
    # Create a safe filename from the target column name
    safe_target_name = target_col.replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = os.path.join(viz_folder, f"{safe_target_name}_distribution.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Target variable visualization saved to '{filename}'")
    
    # 2. Distribution of categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Skip ID columns and columns with too many unique values
    categorical_cols = [col for col in categorical_cols if not (
        'id' in col.lower() or  # Skip ID columns
        'loan_id' in col.lower() or
        '_id' in col.lower() or 
        df[col].nunique() > 20  # Skip columns with too many unique values
    )]
    
    if len(categorical_cols) > 0:
        # Choose a categorical column (prefer 'Sub-Group' if it exists)
        cat_cols_to_plot = []
        if 'Sub-Group' in categorical_cols:
            cat_cols_to_plot.append('Sub-Group')
        elif 'Group' in categorical_cols:
            cat_cols_to_plot.append('Group')
        elif 'Category' in categorical_cols:
            cat_cols_to_plot.append('Category')
        else:
            # Choose up to 3 categorical columns with fewer unique values
            cat_cols_sorted = sorted([(col, df[col].nunique()) for col in categorical_cols], 
                                    key=lambda x: x[1])
            cat_cols_to_plot = [col for col, _ in cat_cols_sorted[:3]]
        
        # Plot each selected categorical column
        for cat_col in cat_cols_to_plot:
            plt.figure(figsize=(10, 8))
            ax = sns.countplot(y=cat_col, data=df, 
                              order=df[cat_col].value_counts().iloc[:20].index)  # Limit to top 20
            plt.title(f"Distribution of {cat_col}")
            # Format the plot to avoid overlap
            ax.tick_params(axis='y', labelsize=9)
            
            safe_col_name = cat_col.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("(", "").replace(")", "")
            filename = os.path.join(viz_folder, f"{safe_col_name}_distribution.png")
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"✅ {cat_col} distribution saved to '{filename}'")
    
    # 3. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    if not numerical_df.empty:
        correlation = numerical_df.corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, mask=mask, cmap="coolwarm", linewidths=0.5, fmt=".2f")
        plt.title(f"{dataset_name} - Correlation Heatmap")
        
        filename = os.path.join(viz_folder, f"{dataset_name}_correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Correlation heatmap saved to '{filename}'")
    
    # 4. Box plots for numerical features
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns[:5]  # First 5 numerical
    if len(numerical_cols) > 0:
        sns.boxplot(data=df[numerical_cols])
        plt.title(f"{dataset_name} - Box Plots of Numerical Features")
        plt.xticks(rotation=45, ha='right')
        
        # Create a descriptive filename
        numerical_desc = "_".join([col.replace(" ", "").replace("/", "")[:5] for col in numerical_cols[:3]])
        filename = os.path.join(viz_folder, f"{dataset_name}_boxplots_{numerical_desc}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Box plots saved to '{filename}'")
    
    # 5. Pair plot for selected numerical features (optional for more detailed view)
    try:
        # Select a few important numerical columns to avoid overcrowding
        important_cols = list(numerical_cols[:3])  # First 3 numerical features
        if target_col not in important_cols and df[target_col].dtype != 'object':
            important_cols.append(target_col)
            
        if len(important_cols) >= 2:  # Need at least 2 columns for a pair plot
            plt.figure(figsize=(10, 8))
            g = sns.pairplot(df[important_cols], height=2.5)
            g.fig.suptitle(f"{dataset_name} - Pair Plot of Key Features", y=1.02)
            
            # Create descriptive filename with feature names
            features_desc = "_".join([col.replace(" ", "").replace("/", "")[:5] for col in important_cols])
            filename = os.path.join(viz_folder, f"{dataset_name}_pairplot_{features_desc}.png")
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"✅ Pair plot saved to '{filename}'")
    except Exception as e:
        print(f"⚠️ Could not create pair plot: {str(e)}")
        
    return viz_folder  # Return the folder name for future reference

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
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, dataset_name, viz_folder):
    """Train and evaluate multiple machine learning models"""
    print("\n🤖 Let's train and evaluate our machine learning models!")
    
    # Create a subfolder for model performance visualizations
    models_folder = os.path.join(viz_folder, "model_results")
    os.makedirs(models_folder, exist_ok=True)
    print(f"✅ Created folder '{models_folder}' for model performance visualizations")
    
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
        "Genetic Algorithm": GeneticAlgorithmClassifier(generations=20),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "AdaBoost": AdaBoostClassifier(),
        "Neural Network": MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,50), early_stopping=True)
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
        
        # Visualize confusion matrix
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=sorted(set(y_test)),
                       yticklabels=sorted(set(y_test)))
            plt.title(f'{dataset_name} - {name} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            safe_name = name.replace(' ', '_').replace('/', '_')
            matrix_file = os.path.join(models_folder, f"{dataset_name}_{safe_name}_confusion_matrix.png")
            plt.savefig(matrix_file, dpi=150)
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not create confusion matrix visualization for {name}: {str(e)}")
    
    # Create a bar chart comparing model accuracies
    try:
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] * 100 for model in model_names]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, accuracies, color='skyblue')
        plt.title(f'{dataset_name} - Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45, ha='right')
        
        # Add accuracy values on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f"{accuracies[i]:.2f}%", ha='center', va='bottom',
                   fontsize=9, rotation=0)
            
        plt.ylim(0, 105)  # Leave space at the top for text
        plt.tight_layout()
        plt.savefig(os.path.join(models_folder, f"{dataset_name}_accuracy_comparison.png"), dpi=150)
        plt.close()
        print(f"✅ Model accuracy comparison chart saved to '{models_folder}/{dataset_name}_accuracy_comparison.png'")
    except Exception as e:
        print(f"⚠️ Could not create accuracy comparison chart: {str(e)}")
    
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
        viz_folder = visualize_data(df, dataset_name)
    except Exception as e:
        print(f"⚠️ Warning: Could not create visualizations: {str(e)}")
        viz_folder = f"{dataset_name}_visualizations"
        os.makedirs(viz_folder, exist_ok=True)
    
    # Step 3: Preprocess the data
    try:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        return None
    
    # Step 4: Train and evaluate models
    try:
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, dataset_name, viz_folder)
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
    
    # Create a folder for comparison results
    comparison_folder = "comparison_results"
    os.makedirs(comparison_folder, exist_ok=True)
    
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
    
    # Get timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    
    # Save comparison to CSV
    csv_file = os.path.join(comparison_folder, f"model_comparison_{timestamp}.csv")
    comparison.to_csv(csv_file)
    print(f"✅ Comparison data saved to '{csv_file}'")
    
    # Create bar chart
    plt.figure(figsize=(14, 10))
    comparison.plot(kind='bar')
    plt.title('Model Performance Comparison Across Datasets (%)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    comparison_chart = os.path.join(comparison_folder, f"all_models_comparison_{timestamp}.png")
    plt.savefig(comparison_chart, dpi=150)
    plt.close()
    
    print(f"✅ Comparison chart saved to '{comparison_chart}'")
    
    # Create a heatmap for easier visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(comparison, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title('Model Performance Comparison Heatmap (%)')
    plt.tight_layout()
    heatmap_file = os.path.join(comparison_folder, f"performance_heatmap_{timestamp}.png")
    plt.savefig(heatmap_file, dpi=150)
    plt.close()
    
    print(f"✅ Performance heatmap saved to '{heatmap_file}'")
    
    return comparison

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