
import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# SMARTPM - MACHINE LEARNING MODEL COMPARISON
# ==========================================

print("=" * 65)
print(" SMARTPM: COMPARING FIVE MACHINE LEARNING ALGORITHMS")
print("=" * 65)

# Create output folder
os.makedirs("model", exist_ok=True)

# 1. LOAD DATASET
print("\n[1/6] Loading dataset...")

df = pd.read_csv("dataset/ai4i2020.csv")

# Remove identifier columns if present
df = df.drop(columns=["UDI", "Product ID"], errors="ignore")

# Encode machine type
df["Type"] = df["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2
})

# Selected machine operating features
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

# Validate dataset columns
required_columns = features + [target]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )

# Remove rows with missing selected values
df = df.dropna(subset=required_columns)

X = df[features]
y = df[target].astype(int)

print(f"Dataset rows: {len(df)}")
print(f"Total input features: {len(features)}")
print(f"Normal samples: {(y == 0).sum()}")
print(f"Failure samples: {(y == 1).sum()}")

# 2. SPLIT DATA
print("\n[2/6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 3. DEFINE FIVE MODELS
# Scaling is included for models sensitive to feature magnitudes.

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ))
    ])
}

# 4. TRAIN AND EVALUATE
print("\n[3/6] Training and evaluating models...\n")

results = []
trained_models = {}
predictions = {}

for name, model in models.items():

    print(f"Training {name}...")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)

    trained_models[name] = model
    predictions[name] = y_pred

    # Failure class is positive class 1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Failure Precision (%)": round(precision * 100, 2),
        "Failure Recall (%)": round(recall * 100, 2),
        "Failure F1 (%)": round(f1 * 100, 2),
        "Training Time (seconds)": round(training_time, 3)
    })

    print(f"  Accuracy:        {accuracy * 100:.2f}%")
    print(f"  Failure precision: {precision * 100:.2f}%")
    print(f"  Failure recall:    {recall * 100:.2f}%")
    print(f"  Failure F1-score:  {f1 * 100:.2f}%")
    print(f"  Training time:     {training_time:.3f} seconds")
    print("-" * 45)

# 5. DISPLAY AND SAVE RESULTS
print("\n[4/6] Preparing comparison table...")

results_df = pd.DataFrame(results)

# Sort by failure-class F1 score, then failure recall
results_df = results_df.sort_values(
    by=["Failure F1 (%)", "Failure Recall (%)"],
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 95)
print("SMARTPM MODEL COMPARISON RESULTS")
print("=" * 95)
print(results_df.to_string(index=False))

results_path = "model/model_comparison_results.csv"
results_df.to_csv(results_path, index=False)

print(f"\nComparison table saved to: {results_path}")

# Select the top model by failure-class F1 score.
# This is an experimental selection, not a guarantee of best
# real-world performance.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

best_model_path = "model/best_comparison_model.pkl"
joblib.dump(best_model, best_model_path)

print(f"\nTop model by failure F1-score: {best_model_name}")
print(f"Saved separately to: {best_model_path}")

# The original model/random_forest_model.pkl is NOT overwritten.

# 6. PLOT MODEL COMPARISON
print("\n[5/6] Creating comparison chart...")

plot_df = results_df.set_index("Model")[
    [
        "Accuracy (%)",
        "Failure Precision (%)",
        "Failure Recall (%)",
        "Failure F1 (%)"
    ]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.78
)

ax.set_title(
    "SmartPM - Machine Learning Model Comparison",
    fontsize=15,
    pad=15
)
ax.set_xlabel("Machine Learning Algorithm")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower right", fontsize=9)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

chart_path = "model/model_comparison.png"
plt.savefig(chart_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Comparison chart saved to: {chart_path}")

# 7. DETAILED REPORTS AND CONFUSION MATRICES
print("\n[6/6] Detailed evaluation reports")

for name in models:
    print("\n" + "=" * 55)
    print(f"MODEL: {name}")
    print("=" * 55)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions[name],
            labels=[0, 1],
            target_names=["Normal", "Failure"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions[name],
        labels=[0, 1]
    ))

print("\n" + "=" * 65)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 65)
print("\nFiles created:")
print("1. model/model_comparison_results.csv")
print("2. model/model_comparison.png")
print("3. model/best_comparison_model.pkl")
print("\nYour original Random Forest model remains unchanged.")
