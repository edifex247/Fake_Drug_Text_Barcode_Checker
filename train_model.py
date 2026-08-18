import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load the drug dataset
data_path = "data/drug_data.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print(df)

# Features used by the machine-learning model
features = [
    "nafdac_valid",
    "batch_valid",
    "barcode_valid",
    "manufacturer_known",
    "text_complete"
]

X = df[features]
y = df["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed.")
print(f"Model accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Create model directory
os.makedirs("model", exist_ok=True)

# Save the trained model
model_path = "model/drug_checker_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully to: {model_path}")