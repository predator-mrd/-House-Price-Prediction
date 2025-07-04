import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("/data_house.csv")
print("Initial shape:", df.shape)

# Step 1: Data Cleaning
df = df.select_dtypes(include=[np.number])  # Drop categorical columns for now
df = df.dropna(axis=1)  # Drop columns with missing values
print("After cleaning:", df.shape)

# Step 2: Feature & Target split
X = df.drop("Price", axis=1)
y = df["Price"]

# Optional: Normalize target
y_log = np.log1p(y)  # log transform to reduce skewness

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "R2": r2}
    print(f"{name} → RMSE: {rmse:.2f}, R2: {r2:.3f}")

# Step 6: Visualize Predictions of Best Model
best_model = models["Random Forest"]
preds = best_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(np.expm1(y_test), np.expm1(preds), alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([0, 800000], [0, 800000], '--', color='red')
plt.grid(True)
plt.show()