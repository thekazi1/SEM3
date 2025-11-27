import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Binarizer

def load_dataset(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(data):
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Encode categorical columns
    for col in data.select_dtypes(include='object').columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # Scaling
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Binarization (FIXED: no more warnings)
    binarizer = Binarizer(threshold=0.5)
    data[numeric_cols] = pd.DataFrame(
        binarizer.fit_transform(data[numeric_cols].values),
        columns=numeric_cols
    )

    return data

def main():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = load_dataset(url)

    if data is not None:
        print("Original Data:")
        print(data.head())

        data = preprocess_data(data)

        print("\nPreprocessed Data:")
        print(data.head())

if __name__ == "__main__":
    main()
