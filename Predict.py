import joblib
import pandas as pd
import logging
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', message='Found unknown categories')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models_and_transformers():
    """Load the trained model and necessary transformers from the models directory."""
    try:
        # Load the ensemble model from models directory
        model = joblib.load('models/model_ensemble.joblib')
        logger.info("Model loaded successfully")

        # Load condition-specific transformers from models directory
        transformers = {}
        conditions = ['DIABETES', 'Depression', 'HTN', 'OA', 'COPD']
        for condition in conditions:
            transformer_path = f'models/{condition}_transformer.joblib'
            transformers[condition] = joblib.load(transformer_path)
            logger.info(f"Transformer for {condition} loaded successfully")

        return model, transformers

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def create_test_patients():
    """Create example high-risk and low-risk test patients."""
    # High-risk patient (elderly, high BMI, high blood pressure, high A1c)
    high_risk_patient = pd.DataFrame({
        'Age_at_Exam': [60],
        'sBP': [160],
        'BMI': [35],
        'LDL': [3.5],
        'HDL': [1.0],
        'A1c': [7.0],
        'TG': [2.5],
        'FBS': [7.5],
        'Total_Cholesterol': [5.5],
        'Sex': ['Male'],
        'Hypertension_Medications': ['AMLODIPINE'],
        'Corticosteroids': ['HYDROCORTISONE']
    })

    # Low-risk patient (young, normal BMI, normal blood pressure, normal A1c)
    low_risk_patient = pd.DataFrame({
        'Age_at_Exam': [25],
        'sBP': [110],
        'BMI': [22],
        'LDL': [2.5],
        'HDL': [1.5],
        'A1c': [5.2],
        'TG': [1.0],
        'FBS': [4.5],
        'Total_Cholesterol': [4.0],
        'Sex': ['Female'],
        'Hypertension_Medications': [None],
        'Corticosteroids': [None]
    })

    return high_risk_patient, low_risk_patient

def make_predictions(models, transformers, patient_data):
    """Make predictions for the given patient data."""
    try:
        predictions = {}
        probabilities = {}
        conditions = ['DIABETES', 'Depression', 'HTN', 'OA', 'COPD']

        for condition in conditions:
            # Transform the data using condition-specific transformer
            X_transformed = transformers[condition].transform(patient_data)

            # Get prediction and probability using the condition-specific model
            condition_model = models[condition]  # Get the specific model for this condition
            pred = condition_model.predict(X_transformed)
            prob = condition_model.predict_proba(X_transformed)

            predictions[condition] = pred[0]
            probabilities[condition] = prob[0][1]  # Probability of positive class

        return predictions, probabilities

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

def format_results(predictions, probabilities):
    """Format the prediction results for display."""
    logger.info("\nPrediction Results:")
    logger.info("-" * 50)

    for condition in predictions.keys():
        result = "Positive" if predictions[condition] else "Negative"
        prob = probabilities[condition] * 100
        logger.info(f"{condition:12} : {result:8} (Risk: {prob:.1f}%)")

def main():
    # Load models and transformers
    model, transformers = load_models_and_transformers()

    # Create test patients
    high_risk_patient, low_risk_patient = create_test_patients()

    # Make predictions for high-risk patient
    logger.info("\nAnalyzing High-Risk Patient:")
    predictions, probabilities = make_predictions(model, transformers, high_risk_patient)
    format_results(predictions, probabilities)

    # Make predictions for low-risk patient
    logger.info("\nAnalyzing Low-Risk Patient:")
    predictions, probabilities = make_predictions(model, transformers, low_risk_patient)
    format_results(predictions, probabilities)

if __name__ == "__main__":
    main()
