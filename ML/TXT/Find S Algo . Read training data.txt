import pandas as pd

def find_s_algorithm(data, target_column):
    # Start with most specific hypothesis
    specific_hypothesis = ['NULL'] * (len(data.columns) - 1)

    for _, row in data.iterrows():
        if row[target_column] == "Yes":
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] == "NULL":
                    specific_hypothesis[i] = row.iloc[i]
                elif specific_hypothesis[i] != row.iloc[i]:
                    specific_hypothesis[i] = "?"
    return specific_hypothesis


def main():
    dataset = {
        'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Sunny'],
        'Temp': ['Warm', 'Warm', 'Cold', 'Warm', 'Warm'],
        'Humidity': ['Normal', 'High', 'High', 'High', 'Normal'],
        'Wind': ['Strong', 'Strong', 'Strong', 'Strong', 'Strong'],
        'Water': ['Warm', 'Warm', 'Cold', 'Warm', 'Warm'],
        'Forecast': ['Same', 'Same', 'Change', 'Same', 'Same'],
        'EnjoySport': ['Yes', 'Yes', 'No', 'Yes', 'Yes']
    }

    df = pd.DataFrame(dataset)
    df.to_csv("training_data.csv", index=False)
    print("Dataset saved to training_data.csv")

    data = pd.read_csv("training_data.csv")
    print("\nTraining Data:")
    print(data)

    specific_hypothesis = find_s_algorithm(data, 'EnjoySport')
    print("\nFinal Specific Hypothesis:")
    print(specific_hypothesis)


if __name__ == "__main__":
    main()
