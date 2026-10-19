import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("dataset/ai4i2020.csv")

# Remove unnecessary ID columns
df = df.drop(["UDI", "Product ID"], axis=1)

# Convert machine Type into numbers
df["Type"] = df["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2
})

# Select only actual machine operating parameters
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Input data
X = df[features]

# Target
y = df["Machine failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("========================================")
print("SMART PREDICTIVE MAINTENANCE SYSTEM")
print("========================================")

print("\nRandom Forest Model Trained Successfully!")

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(model, "model/random_forest_model.pkl")
print("\nModel saved successfully!")