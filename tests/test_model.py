import os
import pandas as pd
import pytest
from fail_prediction import HeartFailureReadmissionPredictor, predict_readmission_risk

data_path = "C:/Users/srishti agarwal/OneDrive/Documents/hackathon/data/mimic-iii"

@pytest.fixture(scope="module")
def predictor():
    model = HeartFailureReadmissionPredictor(data_path)
    model.load_data()
    model.identify_heart_failure_patients()
    model.create_readmission_labels()
    return model

def test_data_loaded(predictor):
    assert predictor.admissions_df is not None
    assert predictor.diagnoses_df.shape[0] > 0

def test_readmission_label_created(predictor):
    assert "READMISSION_30D" in predictor.heart_failure_patients.columns

def test_run_pipeline():
    predictor = HeartFailureReadmissionPredictor(data_path)
    predictor.run_pipeline(data_path)
    assert os.path.exists("C:/Users/srishti agarwal/OneDrive/Documents/hackathon/outputs/best_model.pkl")

def test_prediction_function():
    # Minimal dummy input for the predict function (should match your model's input)
    dummy_data = pd.DataFrame([[65, 5, 1, 0, 2, 1, 0, 0]], columns=[
        'AGE', 'LOS', 'IS_EMERGENCY', 'IS_WEEKEND', 'PREV_ADMISSIONS',
        'INSURANCE_MEDICARE', 'INSURANCE_PRIVATE', 'COMORBIDITY_COUNT'
    ])
    risk = predict_readmission_risk(dummy_data, model_path='C:/Users/srishti agarwal/OneDrive/Documents/hackathon/outputs/best_model.pkl')
    assert 0.0 <= risk[0] <= 1.0
