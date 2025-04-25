# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    try:
        # Load the Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        
        # Explore data structure
        print("\nDataset info:")
        print(df.info())
        
        # Check for missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        # Clean data (though Iris dataset is already clean)
        # For demonstration, we'll show how we would handle missing values
        if df.isnull().sum().sum() > 0:
            df = df.dropna()  # or df.fillna(method='ffill')
            print("\nDataset after handling missing values:")
            print(df.isnull().sum())
        else:
            print("\nNo missing values found in the dataset.")
            
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Task 2: Basic Data Analysis
def perform_data_analysis(df):
    if df is None:
        return
        
    print("\nBasic statistics of numerical columns:")
    print(df.describe())
    
    # Group by species and calculate mean for numerical columns
    print("\nMean values by species:")
    print(df.groupby('species').mean())
    
    # Additional analysis
    print("\nAdditional analysis:")
    print("Average sepal length by species:")
    print(df.groupby('species')['sepal length (cm)'].mean())
    
    # Interesting findings
    print("\nInteresting findings:")
    print("Virginica species has the longest petals on average.")
    print("Setosa has the shortest petals but relatively wide sepals.")

# Task 3: Data Visualization
def create_visualizations(df):
    if df is None:
        return
        
    # Set style for better looking plots
    sns.set(style="whitegrid")
    
    # 1. Line chart (simulating time series by using index as x-axis)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
    plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
    plt.title('Trend of Sepal and Petal Lengths Across Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Length (cm)')
    plt.legend()
    plt.show()
    
    # 2. Bar chart - average sepal length by species
    plt.figure(figsize=(8, 5))
    df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Average Sepal Length (cm)')
    plt.xticks(rotation=0)
    plt.show()
    
    # 3. Histogram - distribution of petal length
    plt.figure(figsize=(8, 5))
    sns.histplot(df['petal length (cm)'], bins=20, kde=True, color='purple')
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.show()
    
    # 4. Scatter plot - sepal length vs petal length
    plt.figure(figsize=(8, 6))
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for species, group in df.groupby('species'):
        plt.scatter(group['sepal length (cm)'], group['petal length (cm)'], 
                   label=species, color=colors[species], alpha=0.7)
    plt.title('Sepal Length vs Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.show()
    
    # Bonus: Pairplot to show all relationships
    sns.pairplot(df, hue='species', palette=colors)
    plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
    plt.show()

# Main execution
if __name__ == "__main__":
    print("=== Task 1: Load and Explore the Dataset ===")
    iris_df = load_and_explore_data()
    
    print("\n=== Task 2: Basic Data Analysis ===")
    perform_data_analysis(iris_df)
    
    print("\n=== Task 3: Data Visualization ===")
    create_visualizations(iris_df)