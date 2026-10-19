
import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("dataset/ai4i2020.csv")

df = df.drop(
    columns=["UDI", "Product ID"],
    errors="ignore"
)

df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

# 2. Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Train the original Random Forest configuration
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Get estimated probability of machine failure
failure_index = list(model.classes_).index(1)
failure_probabilities = model.predict_proba(X_test)[:, failure_index]

# 5. Test different decision thresholds
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

results = []

for threshold in thresholds:
    predictions = (
        failure_probabilities >= threshold
    ).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "Failure Precision (%)": round(
            precision_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "Failure Recall (%)": round(
            recall_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "Failure F1 (%)": round(
            f1_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "False Alarms": int(
            confusion_matrix(y_test, predictions, labels=[0, 1])[0, 1]
        ),
        "Missed Failures": int(
            confusion_matrix(y_test, predictions, labels=[0, 1])[1, 0]
        )
    })

results_df = pd.DataFrame(results)

os.makedirs("model", exist_ok=True)

results_df.to_csv(
    "model/threshold_results.csv",
    index=False
)

print("\nTHRESHOLD COMPARISON")
print(results_df.to_string(index=False))

# 6. Plot precision and recall
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["Threshold"],
    results_df["Failure Precision (%)"],
    marker="o",
    label="Failure Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Failure Recall (%)"],
    marker="o",
    label="Failure Recall"
)

plt.xlabel("Probability Threshold")
plt.ylabel("Percentage (%)")
plt.title("Random Forest: Failure Precision vs Recall")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model/threshold_comparison.png",
    dpi=150
)

plt.close()

# 7. Save the model separately.
# The threshold is stored separately; the original model is untouched.
joblib.dump(
    {
        "model": model,
        "threshold": 0.50,
        "features": features
    },
    "model/threshold_test_model.pkl"
)

print("\nSaved results: model/threshold_results.csv")
print("Saved chart: model/threshold_comparison.png")
print("Saved test model: model/threshold_test_model.pkl")
print("Original model was not modified.")
