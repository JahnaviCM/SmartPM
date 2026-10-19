
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
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

DATA_PATH = "dataset/ai4i2020.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Run this file from your project root."
    )

df = pd.read_csv(DATA_PATH)

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

# --------------------------------------------------
# 2. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. TEST DIFFERENT RANDOM FOREST SETTINGS
# --------------------------------------------------

experiments = {
    "Original Random Forest": {
        "n_estimators": 100,
        "class_weight": None,
        "max_depth": None
    },
    "Balanced Random Forest": {
        "n_estimators": 200,
        "class_weight": "balanced",
        "max_depth": None
    },
    "Balanced Shallow Forest": {
        "n_estimators": 200,
        "class_weight": "balanced",
        "max_depth": 15
    }
}

results = []
trained_models = {}

for name, settings in experiments.items():
    print("\nTraining:", name)

    model = RandomForestClassifier(
        **settings,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Failure Precision (%)": round(precision * 100, 2),
        "Failure Recall (%)": round(recall * 100, 2),
        "Failure F1 (%)": round(f1 * 100, 2),
        "Training Time (seconds)": round(training_time, 3)
    })

    trained_models[name] = model

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Failure precision:", round(precision * 100, 2), "%")
    print("Failure recall:", round(recall * 100, 2), "%")
    print("Failure F1:", round(f1 * 100, 2), "%")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Failure"],
        zero_division=0
    ))

# --------------------------------------------------
# 4. SAVE COMPARISON RESULTS
# --------------------------------------------------

os.makedirs("model", exist_ok=True)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Failure F1 (%)",
    ascending=False
)

results_df.to_csv(
    "model/improved_model_results.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 5. SAVE THE BEST MODEL SEPARATELY
# --------------------------------------------------

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(
    best_model,
    "model/improved_random_forest_model.pkl"
)

print("\nBest model by failure F1:", best_name)
print("Saved: model/improved_random_forest_model.pkl")
print("Results: model/improved_model_results.csv")

# --------------------------------------------------
# 6. PLOT COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Failure Recall (%)"]
)

plt.ylabel("Failure Recall (%)")
plt.xlabel("Model")
plt.title("Random Forest Failure Detection Comparison")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig("model/improved_model_comparison.png", dpi=150)
plt.close()

print("Chart: model/improved_model_comparison.png")
print("\nImprovement experiment completed successfully!")
