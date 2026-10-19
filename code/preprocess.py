import pandas as pd

# Load dataset
df = pd.read_csv("dataset/ai4i2020.csv")

# Remove unnecessary columns
df = df.drop(["UDI", "Product ID"], axis=1)

# Convert machine type into numbers
df["Type"] = df["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2
})

print("Preprocessing completed!")

print("\nDataset after preprocessing:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)
# Input features
X = df.drop("Machine failure", axis=1)

# Target variable
y = df["Machine failure"]

print("\nInput Features:")
print(X.columns)

print("\nTarget:")
print("Machine failure")

print("\nX shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)