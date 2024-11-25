import sys
import pandas as pd
import joblib  # Import joblib for loading the saved model

# Function to load model and make prediction
def predict(year, district, population, week, rainsum, meantemperature, model_path):
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
    model_path = r'C:\Users\hp\Desktop\weekly hospitalized cases\hospital-prediction\trained_model.pkl'
    
    # Check if model exists
    try:
        year, district, population, week, rainsum, meantemperature = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    except IndexError:
        print("Error: Missing command-line arguments. Expected: year, district, population, week, rainsum, meantemperature")
        sys.exit(1)
        
    # Use the model for prediction
    prediction = predict(year, district, population, week, rainsum, meantemperature, model_path)
    print(f"Predicted weekly hospitalized cases: {prediction}")
