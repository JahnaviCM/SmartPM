import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("dataset/ai4i2020.csv")

# Remove unnecessary columns
df = df.drop(["UDI", "Product ID"], axis=1)

# Convert Type
df["Type"] = df["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2
})

# Features used by our model
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

# Same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load("model/random_forest_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Failure"]
)

disp.plot()

plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix.png")
plt.show()

# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()

print("\nGraphs created successfully!")