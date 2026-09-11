import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("data/iris.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model parameter
n_estimators = 100

# Start MLflow experiment
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, name="model")

    print("Model trained successfully!")
    print("Accuracy:", accuracy)