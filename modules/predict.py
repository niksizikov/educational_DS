import os
import json
import dill
import pandas as pd

def load_model(model_path):
    """Loads the trained model from a .pkl file."""
    with open(model_path, 'rb') as file:
        model = dill.load(file)
        return model


def load_test_data(test_folder):
    """Loads all JSON files from the test folder and combines them into a DataFrame."""
    test_dataframes = []
    for file_name in os.listdir(test_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(test_folder, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                test_dataframes.append(pd.DataFrame([data]))
    if not test_dataframes:
        raise ValueError(f"No files to process in the folder: {test_folder}")
    return pd.concat(test_dataframes, ignore_index=True)


def save_predictions(predictions, output_path):
    """Saves predictions to a CSV file."""
    predictions.to_csv(output_path, index=False)
    print(f"Predictions have been saved to {output_path}")


def predict():
    """Main function to load the model, make predictions, and save them."""
    # Paths
    # path = os.environ.get('PROJECT_PATH', '.')
    model_path = 'data/models/cars_pipe_202412101252.pkl'
    test_folder = 'data/test'
    output_file = 'data/predictions/predictions.csv'

    # Load the model
    print("Loading the model...")
    model = load_model(model_path)
    print(f"Model successfully loaded: {type(model).__name__}")

    # Load test data
    print("Loading test data...")
    test_data = load_test_data(test_folder)
    print(f"Test data loaded: {test_data.shape[0]} rows")

    # Perform predictions
    print("Performing predictions...")
    test_data['price_category'] = model.predict(test_data)
    print("Predictions completed.")

    # Save predictions
    print("Saving predictions...")
    save_predictions(test_data, output_file)


if __name__ == '__main__':
    predict()
