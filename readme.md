# Multi-Condition Health Risk Prediction Model

An advanced machine learning ensemble model for predicting multiple health conditions using patient health data. The model uses a voting classifier combining Random Forest, XGBoost, and LightGBM with SMOTE balancing to handle class imbalance.

## Conditions Predicted

The model predicts risk for five health conditions:
- Diabetes
- Depression
- Hypertension (HTN)
- Osteoarthritis (OA)
- COPD

## Features

The model analyzes the following patient characteristics:

- **Core Health Metrics**:
  - Fasting Blood Sugar (FBS)
  - HbA1c levels
  - Blood Pressure (sBP)
  - BMI
  - Lipid Profile (HDL, LDL, TG, Total Cholesterol)
  - Age at Examination
  - Sex

- **Medical History**:
  - Hypertension Medications
  - Corticosteroid Usage

- **Engineered Features**:
  - Metabolic Risk Score
  - Age-based Risk Groups
  - BMI-A1c Interactions
  - Combined Risk Factors

## Requirements
```
numpy
pandas
scikit-learn>=1.0
xgboost
lightgbm
imbalanced-learn
joblib
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/MultiConditionHealthPredictor.git
   cd MultiConditionHealthPredictor
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Train new models using your dataset:
```bash
python Model.py
```

This will:
- Load and preprocess the data
- Create condition-specific feature transformers
- Train ensemble models for each condition
- Save models and transformers in the `models/` directory
- Generate detailed performance metrics

### Making Predictions

Use the trained models for predictions:
```bash
python Predict.py
```

The script includes example high-risk and low-risk patient profiles and outputs risk predictions for all conditions.

## Model Architecture

- **Ensemble Approach**: Combines predictions from:
  - Random Forest Classifier
  - XGBoost Classifier
  - LightGBM Classifier

- **Data Processing**:
  - Condition-specific feature engineering
  - SMOTE balancing for rare conditions
  - Automated handling of missing values
  - Feature scaling and encoding

## Performance

Latest model performance metrics for each condition:

- **Diabetes**:
  - Accuracy: 88%
  - ROC AUC: 0.92

- **Depression**:
  - Accuracy: 85%
  - ROC AUC: 0.89

- **Hypertension**:
  - Accuracy: 87%
  - ROC AUC: 0.91

*(Similar metrics available for OA and COPD)*

## Files Description

- `Model.py`: Main training script with ensemble model implementation
- `predict.py`: Prediction interface with example patients
- `models/`: Directory containing:
  - Trained models (not included in repository due to size)
  - Feature transformers
  - Model statistics
- `requirements.txt`: Required Python packages

Note: The trained model ensemble files are not included in this repository due to size limitations. To generate these files, simply run `Model.py` with your dataset.

## Apple Silicon Optimization

The model automatically enables optimized frameworks on Apple Silicon processors when scikit-learn 1.0+ is available.

## Logging

Comprehensive logging in `diabetes_model.log` includes:
- Training metrics for each condition
- Feature importance rankings
- Error tracking
- Prediction results

## Future Development

- Web API for real-time predictions
- Additional health conditions
- Integration with electronic health records
- Mobile application interface
- Expanded feature engineering
