import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("car_data.csv")

# Clean Price column
data["Price"] = data["Price"].astype(str).str.replace(",", "")
data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

# Clean kms_driven column
data["kms_driven"] = data["kms_driven"].astype(str).str.replace(" kms", "")
data["kms_driven"] = data["kms_driven"].str.replace(",", "")
data["kms_driven"] = pd.to_numeric(data["kms_driven"], errors="coerce")

# Clean year column
data["year"] = pd.to_numeric(data["year"], errors="coerce")

# Remove missing values
data = data.dropna()

# Convert categorical columns
data = pd.get_dummies(data, columns=["name", "company", "fuel_type"])

# Features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("Model Accuracy:", round(score * 100, 2), "%")

# Sample prediction
sample = X_test.iloc[[0]]
predicted_price = model.predict(sample)

print("Predicted Car Price:", round(predicted_price[0], 2))