import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/random_forest_model.pkl")

print("======================================")
print(" SMART PREDICTIVE MAINTENANCE SYSTEM ")
print("======================================")

# Get machine details from user
machine_type = input("Enter Machine Type (L/M/H): ").upper()

air_temperature = float(input("Enter Air Temperature [K]: "))
process_temperature = float(input("Enter Process Temperature [K]: "))
rotational_speed = float(input("Enter Rotational Speed [rpm]: "))
torque = float(input("Enter Torque [Nm]: "))
tool_wear = float(input("Enter Tool Wear [min]: "))

# Convert machine type
type_value = {
    "L": 0,
    "M": 1,
    "H": 2
}

machine_type = type_value[machine_type]

# Create input data
input_data = pd.DataFrame([[
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
]], columns=[
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
])

# Make prediction
prediction = model.predict(input_data)[0]

print("\n======================================")

if prediction == 0:
    print("Prediction: MACHINE NORMAL")
    print("Maintenance is not immediately required.")
else:
    print("Prediction: MACHINE FAILURE")
    print("Maintenance is recommended.")

print("======================================")