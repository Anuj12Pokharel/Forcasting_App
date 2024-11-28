import sys
import pandas as pd
import joblib
import os

# Function to load model and make prediction
def predict(year, district, population, week, rainsum, meantemperature, model_path):
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load the trained model
    model = joblib.load(model_path)

    # Prepare input data for prediction
    input_data = pd.DataFrame([[int(year), int(district), int(population), int(week), float(rainsum), float(meantemperature)]],
                              columns=['year', 'district', 'population', 'week', 'rainsum', 'meantemperature'])

    # Make prediction
    predicted_cases = model.predict(input_data)
    return predicted_cases[0]

# Main entry point
if __name__ == "__main__":
    # Define file paths
    model_path = r'C:\Users\hp\Desktop\weekly hospitalized\hospital-prediction\trained_model.pkl'
    
    # Check if all required command-line arguments are provided
    if len(sys.argv) != 7:
        print("Error: Missing command-line arguments.")
        print("Usage: python predict_model.py <year> <district> <population> <week> <rainsum> <meantemperature>")
        sys.exit(1)

    # Extract command-line arguments
    try:
        year, district, population, week, rainsum, meantemperature = sys.argv[1:7]
    except Exception as e:
        print(f"Error processing input arguments: {e}")
        sys.exit(1)

    # Make prediction
    try:
        prediction = predict(year, district, population, week, rainsum, meantemperature, model_path)
        print(f"Predicted weekly hospitalized cases: {prediction}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
