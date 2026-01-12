import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import numpy as np

# --- 1. Train the model on Iris dataset ---
print("1. Loading and preparing Iris dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("2. Training Logistic Regression model...")
model = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 2. Save the model in pkl file ---
model_filename_pkl = 'iris_logistic_regression_model_integrated.pkl'
print(f"3. Saving model to '{model_filename_pkl}'...")
joblib.dump(model, model_filename_pkl)
print("Model saved successfully.")

# --- 3. Perform 3 tests on it ---
# These tests will be written to a temporary file and run by pytest
test_script_content = f"""
import pytest
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import numpy as np

# Define the filename for the saved model (must match the one used above)
model_filename = '{model_filename_pkl}'

# This check is crucial if the script were run standalone, but here it acts as a safeguard
if not os.path.exists(model_filename):
    pytest.fail(f"Model file '{{model_filename}}' not found. Cannot run tests.")

# Load the trained model and data for tests
loaded_model = joblib.load(model_filename)
iris = load_iris()
X_data = pd.DataFrame(iris.data, columns=iris.feature_names)
y_data = pd.Series(iris.target)
_, X_test_data, _, y_test_data = train_test_split(X_data, y_data, test_size=0.3, random_state=42, stratify=y_data)

@pytest.fixture(scope='module')
def trained_model():
    return loaded_model

@pytest.fixture(scope='module')
def test_data():
    return X_test_data, y_test_data

def test_accuracy_above_80_percent(trained_model, test_data):
    \"\"\"
    Test that the model's accuracy on the test set is above 80%.
    \"\"\"
    X_test, y_test = test_data
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.80, f"Model accuracy ({{accuracy:.2f}}) is not above 80%."

def test_prediction_shape_matches_y_test(trained_model, test_data):
    \"\"\"
    Test to check if the shape of the model's predictions matches the shape of y_test.
    \"\"\"
    X_test, y_test = test_data
    y_pred = trained_model.predict(X_test)
    assert y_pred.shape == y_test.shape, f"Prediction shape {{y_pred.shape}} does not match y_test shape {{y_test.shape}}"

def test_accuracy_not_exceed_100_percent(trained_model, test_data):
    \"\"\"
    Test that the model's accuracy on the test set does not exceed 100% (1.0).
    \"\"\"
    X_test, y_test = test_data
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy <= 1.0, f"Model accuracy ({{accuracy:.2f}}) exceeded 100% (1.0)."
"""

# Write the test script content to a temporary file
test_filename = 'iris_model_integrated_tests.py'
with open(test_filename, 'w') as f:
    f.write(test_script_content)

print(f"4. Running 3 tests from '{test_filename}'...")
# Run pytest on the temporary file
os.system(f"pytest {test_filename}")

# Clean up temporary files
if os.path.exists(model_filename_pkl):
    os.remove(model_filename_pkl)
    print(f"Cleaned up '{model_filename_pkl}'.")
if os.path.exists(test_filename):
    os.remove(test_filename)
    print(f"Cleaned up '{test_filename}'.")
