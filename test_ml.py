import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Sample data for testing
sample_data = {
    'workclass': ['Private', 'Self-emp', 'Private'],
    'education': ['Bachelors', 'Masters', 'Doctorate'],
    'marital-status': ['Never-married', 'Married', 'Divorced'],
    'occupation': ['Tech-support', 'Sales', 'Other-service'],
    'relationship': ['Not-in-family', 'Husband', 'Not-in-family'],
    'race': ['White', 'Black', 'White'],
    'sex': ['Male', 'Female', 'Male'],
    'native-country': ['United-States', 'United-States', 'United-States'],
    'salary': [0, 1, 0]
}
sample_df = pd.DataFrame(sample_data)

# Prepare sample data
X_sample, y_sample, encoder, lb = process_data(
    sample_df,
    categorical_features=[
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ],
    label="salary",
    training=True
)

X_test_sample, y_test_sample, _, _ = process_data(
    sample_df,
    categorical_features=[
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ],
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

model = RandomForestClassifier(random_state=90).fit(X_sample, y_sample)

def test_inference_type():
    """
    Test if the inference function returns a numpy array.
    """
    preds = inference(model, X_test_sample)
    assert isinstance(preds, np.ndarray), "Inference results should be a numpy array."

def test_model_algorithm():
    """
    Test if the model is using the RandomForestClassifier algorithm.
    """
    assert isinstance(model, RandomForestClassifier), "The model should be an instance of RandomForestClassifier."

def test_compute_model_metrics():
    """
    Test if the compute_model_metrics function returns precision, recall, and fbeta as floats.
    """
    preds = inference(model, X_test_sample)
    precision, recall, fbeta = compute_model_metrics(y_test_sample, preds)
    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(fbeta, float), "Fbeta score should be a float."
