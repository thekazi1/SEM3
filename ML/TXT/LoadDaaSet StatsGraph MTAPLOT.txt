import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_url):
    try:
        data = pd.read_csv(file_url)
        print("Dataset loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def descriptive_statistics(data):
    print("\nDescriptive Statistics:")
    print(data.describe(include='all'))
    print("\nMissing Values in Each Column:")
    print(data.isnull().sum())

def create_visualizations(data):
    sns.set(style="whitegrid")

    numeric_data = data.select_dtypes(include='number')

    # Histogram
    numeric_data.hist(bins=30, edgecolor='black', figsize=(12, 10))
    plt.suptitle('Histogram of Numeric Features')
    plt.show()

    # Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numeric_data)
    plt.title('Box Plot of Numeric Features')
    plt.xticks(rotation=45)
    plt.show()

    # Heatmap
    plt.figure(figsize=(10, 8))
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # Pairplot
    selected_cols = ['Age', 'Fare', 'Pclass']
    sns.pairplot(data[selected_cols].dropna())
    plt.suptitle('Pairplot of Numeric Features', y=1.02)
    plt.show()

def main():
    file_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = load_dataset(file_url)

    if data is not None:
        descriptive_statistics(data)
        create_visualizations(data)

if __name__ == "__main__":
    main()
