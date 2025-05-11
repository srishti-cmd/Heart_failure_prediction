# Heart Failure Readmission Prediction Model

## Overview
This model predicts the risk of 30-day readmission for heart failure patients.

## Files
- `best_model.pkl`: The trained prediction model
- `feature_names.pkl`: Names of features used by the model
- Several visualization files showing model performance

## Usage
```python
import pickle

# Load the model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Make predictions on new data
# Note: New data should have the same structure as training data
predictions = model.predict_proba(new_patient_data)[:, 1]
```
