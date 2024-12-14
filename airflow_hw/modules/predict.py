import os
import json
import dill
import pandas as pd


def load_model(model_folder):
    """
    Finds the most recent model in the specified folder and loads it.
    """
    # Find all .pkl files in the folder
    model_files = [
        os.path.join(model_folder, f) for f in os.listdir(model_folder)
        if f.endswith('.pkl')
    ]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_folder}")

    # Find the most recently modified file
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading the latest model: {latest_model_path}")

    # Load the model
    with open(latest_model_path, 'rb') as file:
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
    path = os.environ.get('PROJECT_PATH', '.')
    model_folder = os.path.join(path, 'data/models')
    test_folder = os.path.join(path, 'data/test')
    output_file = os.path.join(path, 'data/predictions/predictions.csv')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the latest model
    print("Loading the latest model...")
    model = load_model(model_folder)
    print(f"Model successfully loaded: {type(model).__name__}")

    # Load test data
    print("Loading test data...")
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"Test folder does not exist: {test_folder}")
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
