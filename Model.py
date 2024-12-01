import pandas as pd
import platform
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diabetes_model.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Apple Silicon optimizations
try:
    import sklearn
    if sklearn.__version__ >= "1.0":
        try:
            from sklearn import config_context
            has_apple_config = True
            logger.info("Apple optimized frameworks available (scikit-learn %s)", sklearn.__version__)
        except ImportError:
            # Fallback for older sklearn versions
            from sklearn.utils import _config
            config_context = _config.config_context
            has_apple_config = True
            logger.info("Apple optimized frameworks available (scikit-learn %s)", sklearn.__version__)
    else:
        has_apple_config = False
        logger.warning("scikit-learn version %s detected. Version 1.0+ required for Apple optimizations", sklearn.__version__)
except ImportError as e:
    has_apple_config = False
    logger.warning("Apple optimized frameworks not available: %s", str(e))

# Check if running on Apple Silicon
is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
logger.info("Platform: %s, Processor: %s, Is Apple Silicon: %s",
            platform.system(), platform.processor(), is_apple_silicon)

def create_diabetes_features(X):
    """Create diabetes-specific feature interactions."""
    X = X.copy()

    # Basic interactions (only for numeric features)
    if all(col in X.columns for col in ['BMI', 'A1c', 'FBS', 'Age_at_Exam']):
        # Create interactions with clean names
        X['BMI_x_A1c'] = X['BMI'] * X['A1c']
        X['Age_x_A1c'] = X['Age_at_Exam'] * X['A1c']
        X['BMI_x_FBS'] = X['BMI'] * X['FBS']

        # Risk factors combination
        X['metabolic_risk'] = X['BMI'] * X['A1c'] * X['FBS'] / 100

        # Age-based risk groups with numeric values
        age_bins = [0, 40, 60, 100]
        X['age_risk'] = pd.cut(X['Age_at_Exam'],
                              bins=age_bins,
                              labels=[1, 2, 3])
        # Fill NaN values with median (2)
        X['age_risk'] = X['age_risk'].fillna(2).astype(int)

        # Additional risk factors
        X['high_bmi_flag'] = (X['BMI'] > 30).astype(int)
        X['high_fbs_flag'] = (X['FBS'] > 5.6).astype(int)
        X['high_a1c_flag'] = (X['A1c'] > 5.7).astype(int)

        # Combined risk score
        X['risk_score'] = X['high_bmi_flag'] + X['high_fbs_flag'] + X['high_a1c_flag']

    # Clean all column names - make them very simple
    X.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                 .replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt')
                 .replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                 for col in X.columns]

    return X

def load_and_preprocess_data(file_path):
    """Load and preprocess the diabetes data with enhanced feature engineering."""
    global condition_specific_transformers
    condition_specific_transformers = {}
    # Load data
    data = pd.read_csv(file_path)

    # Convert 'DIABETES' YES/NO to 1/0 if needed
    data['DIABETES'] = data['DIABETES'].map({'Yes': 1, 'No': 0})


    # Define feature columns - add sBP to numeric features
    numeric_features = ['Age_at_Exam', 'sBP', 'BMI', 'LDL', 'HDL', 'A1c',
                       'TG', 'FBS', 'Total_Cholesterol']

    # Modify categorical features based on prediction target
    base_categorical_features = ['Sex']
    medication_features = {
        'Depression': ['Hypertension_Medications', 'Corticosteroids'],
        'HTN': ['Corticosteroids'],  # Removed Hypertension_Medications
        'OA': ['Hypertension_Medications', 'Corticosteroids'],
        'COPD': ['Hypertension_Medications', 'Corticosteroids'],
        'DIABETES': ['Hypertension_Medications', 'Corticosteroids']
    }

    # Target columns
    target_columns = ['Depression', 'HTN', 'OA', 'COPD', 'DIABETES']

    # Drop date columns and unnecessary columns before processing
    columns_to_drop = [
        'sBP_Date', 'BMI_Date', 'LDL_Date', 'HDL_Date', 'A1c_Date',
        'TG_Date', 'FBS_Date', 'Total_Cholesterol_Lab_Date',
        'DM_OnsetDate', 'Depression_OnsetDate', 'HTN_OnsetDate',
        'OA_OnsetDate', 'COPD_Date', 'Hypertension_Medications_First_Instance',
        'Corticosteroids_first_instance', 'leastO(A1c_Date)',
        'leastO(DM_OnsetDate)', 'leastO(FBS_Date)', 'LeastOfAll',
        'A1C_BEF_DM', 'FBS_BEF_DM', 'Patient_ID', 'DM_Onset_Revised',
        'DM_Onset_Revised_1YrPrior', 'FBS>DM',
        'Diabetes'
    ]

    data = data.drop(columns=columns_to_drop, errors='ignore')
    # Create condition-specific transformers and features
    condition_specific_features = {}
    condition_specific_transformers = {}

    for condition in target_columns:
        # Combine base categorical features with condition-specific medication features
        categorical_features = base_categorical_features + medication_features[condition]
        condition_specific_features[condition] = numeric_features + categorical_features

        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # Combine transformers - Make sure 'sBP' is included in numeric_features
        condition_specific_transformers[condition] = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),  # 'sBP' should be included here
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
    # Process data for each condition
    X_transformed = {}
    for condition in target_columns:
        # Prepare X for this condition
        X = data[condition_specific_features[condition]]

        # Apply preprocessing
        X_transformed_array = condition_specific_transformers[condition].fit_transform(X)

        # Get feature names - Make sure to include 'sBP' in the output
        numeric_features_out = numeric_features  # This should include 'sBP'
        categorical_features_out = condition_specific_transformers[condition].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            condition_specific_features[condition][len(numeric_features):])

        # Clean feature names
        feature_names = numeric_features_out + list(categorical_features_out)
        feature_names = [name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                        .replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt')
                        .replace(',', '').replace('/', '_').replace('\\', '_').replace('.', '_')
                        for name in feature_names]

        # Create DataFrame
        X_transformed[condition] = pd.DataFrame(X_transformed_array, columns=feature_names)

        # Add diabetes-specific features
        X_transformed[condition] = create_diabetes_features(X_transformed[condition])

    return X_transformed, data[target_columns]

def train_model(X_dict, y):
    """Train the model with condition-specific features."""
    models = {}
    model_stats = {}  # New dictionary to store model statistics

    for condition in y.columns:
        logger.info(f"\nTraining model for {condition}")

        # Get condition-specific features
        X = X_dict[condition]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y[condition], test_size=0.2, random_state=42)

        # Calculate class distribution with safety checks
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)

        if n_pos == 0 or n_neg == 0:
            logger.warning(f"Warning: {condition} has a class with zero samples. Using auto sampling.")
            sampling_strategy = 'auto'
        else:
            n_minority = min(n_pos, n_neg)
            n_majority = max(n_pos, n_neg)
            ratio = n_minority / n_majority

            # If ratio is too small, use a fixed strategy
            if ratio < 0.2:
                sampling_strategy = 0.5  # Oversample minority class to be 50% of majority
            else:
                sampling_strategy = 'auto'  # Let SMOTE decide based on the data

        # Create base models with balanced class weights
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )

        # Safely calculate scale_pos_weight for each condition
        def get_scale_pos_weight(y_col):
            n_neg = len(y_train[y_train == 0])
            n_pos = len(y_train[y_train == 1])
            return n_neg / max(n_pos, 1)  # Avoid division by zero

        # Create condition-specific parameters
        model_params = {
            'xgb': XGBClassifier(
                scale_pos_weight=get_scale_pos_weight(y_train),
                random_state=42
            ),
            'rf': clone(rf),
            'lgbm': LGBMClassifier(
                class_weight='balanced',
                random_state=42
            )
        }

        # Create a voting classifier with condition-specific models
        estimators = [
            ('rf', model_params['rf']),
            ('xgb', model_params['xgb']),
            ('lgbm', model_params['lgbm'])
        ]

        # Create SMOTE with appropriate sampling strategy
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

        model = Pipeline([
            ('smote', smote),
            ('voting', VotingClassifier(estimators=estimators, voting='soft'))
        ])

        # Fit the model
        model.fit(X_train, y_train)
        models[condition] = model

        # Log class distribution
        logger.info(f"Class distribution for {condition}:")
        logger.info(f"Negative cases: {len(y_train[y_train == 0])}")
        logger.info(f"Positive cases: {len(y_train[y_train == 1])}")

        # Rest of the evaluation code remains the same
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        logger.info(f"\nMetrics for {condition}:")
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        logger.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.3f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        if hasattr(model.named_steps['voting'].estimators_[0], 'feature_importances_'):
            importances = analyze_feature_importance(
                model.named_steps['voting'].estimators_[0],
                X.columns
            )
            logger.info(f"\nTop 10 features for {condition}:\n{importances.head(10)}")

        # Store model statistics
        model_stats[condition] = {
            'cv_scores': {
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            },
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        if hasattr(model.named_steps['voting'].estimators_[0], 'feature_importances_'):
            importances = analyze_feature_importance(
                model.named_steps['voting'].estimators_[0],
                X.columns
            )
            model_stats[condition]['top_features'] = importances.head(10).to_dict('records')

    return models, model_stats  # Return both models and stats

def analyze_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        return importances.sort_values('importance', ascending=False)
    return None

def main():
    # Create models directory if it doesn't exist
    import os
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Load and preprocess data
    X, y = load_and_preprocess_data('diabetes_data.csv')

    # Train model and get stats
    model, model_stats = train_model(X, y)

    # Save models and transformers to models directory
    joblib.dump(model, os.path.join(models_dir, 'model_ensemble.joblib'))
    logger.info("Model saved as models/model_ensemble.joblib")

    # Save condition-specific components
    for condition, transformer in condition_specific_transformers.items():
        # Save transformer
        transformer_path = os.path.join(models_dir, f'{condition}_transformer.joblib')
        joblib.dump(transformer, transformer_path)
        logger.info(f"Transformer for {condition} saved as {transformer_path}")

        # Save scaler
        numeric_scaler = transformer.named_transformers_['num'].named_steps['scaler']
        scaler_path = os.path.join(models_dir, f'{condition}_scaler.joblib')
        joblib.dump(numeric_scaler, scaler_path)
        logger.info(f"Scaler for {condition} saved as {scaler_path}")

    # Print detailed model statistics
    logger.info("\n=== Model Performance Summary ===")
    for condition, stats in model_stats.items():
        logger.info(f"\n{condition} Model Performance:")
        logger.info(f"Cross-validation: {stats['cv_scores']['mean']:.3f} (±{stats['cv_scores']['std']*2:.3f})")
        logger.info(f"Test Accuracy: {stats['accuracy']:.3f}")
        logger.info(f"ROC AUC Score: {stats['roc_auc']:.3f}")

        logger.info("\nClassification Report:")
        for class_label, metrics in stats['classification_report'].items():
            if isinstance(metrics, dict):
                logger.info(f"Class {class_label}:")
                logger.info(f"  Precision: {metrics['precision']:.3f}")
                logger.info(f"  Recall: {metrics['recall']:.3f}")
                logger.info(f"  F1-score: {metrics['f1-score']:.3f}")

        if 'top_features' in stats:
            logger.info("\nTop 10 Important Features:")
            for idx, feature in enumerate(stats['top_features'], 1):
                logger.info(f"{idx}. {feature['feature']}: {feature['importance']:.4f}")

    # Save model statistics to file
    import json
    stats_path = os.path.join(models_dir, 'model_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(model_stats, f, indent=4)
    logger.info(f"\nModel statistics saved to {stats_path}")

def make_predictions(models, transformers, patient_data):
    try:
        predictions = {}
        probabilities = {}
        conditions = ['DIABETES', 'Depression', 'HTN', 'OA', 'COPD']

        # Ensure medications match training data (example medications from your dataset)
        known_hypertension_meds = [
            'AMLODIPINE',
            'BISOPROLOL',
            'HYDROCHLOROTHIAZIDE',
            'RAMIPRIL',
            'LOSARTAN',
            'METOPROLOL',
            'NIFEDIPINE',
            'DILTIAZEM'
        ]
        known_corticosteroids = [
            'HYDROCORTISONE',
            'DEXAMETHASONE',
            'METHYLPREDNISOLONE'
        ]

        # Clean medication fields to use only known categories
        if 'Hypertension_Medications' in patient_data.columns:
            patient_data['Hypertension_Medications'] = patient_data['Hypertension_Medications'].apply(
                lambda x: next((med for med in known_hypertension_meds if med in str(x)), None)
            )

        if 'Corticosteroids' in patient_data.columns:
            patient_data['Corticosteroids'] = patient_data['Corticosteroids'].apply(
                lambda x: next((med for med in known_corticosteroids if med in str(x)), None)
            )

        for condition in conditions:
            # Transform the data using condition-specific transformer
            X_transformed = transformers[condition].transform(patient_data)

            # Get feature names from transformer
            feature_names = transformers[condition].get_feature_names_out()

            # Convert to DataFrame with proper feature names
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

            # Make prediction
            prediction = models[condition].predict(X_transformed)
            probability = models[condition].predict_proba(X_transformed)

            predictions[condition] = prediction[0]
            probabilities[condition] = probability[0][1]  # Probability of positive class

        return predictions, probabilities

    except Exception as e:
        logger.error(f"Error in make_predictions: {str(e)}")
        raise

if __name__ == "__main__":
    main()
