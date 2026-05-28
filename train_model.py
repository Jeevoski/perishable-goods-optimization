# train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Initializing training sequence...")

# 1. Simulate Historical Data (In production, this is queried from a Data Warehouse)
# Features: Hours to Expiry, Current Stock, Competitor Price
# Target: Optimal Discount Percentage to clear inventory
np.random.seed(42)
data_size = 10000
df = pd.DataFrame({
    'hours_to_expiry': np.random.uniform(1, 72, data_size),
    'current_stock': np.random.randint(1, 50, data_size),
    'competitor_price': np.random.uniform(2.0, 10.0, data_size),
})
# Synthetic logic: Shorter expiry + higher stock = higher required discount
df['optimal_discount'] = (72 / df['hours_to_expiry']) * (df['current_stock'] * 0.5)
df['optimal_discount'] = df['optimal_discount'].clip(0, 75) # Cap discount at 75%

# 2. Split Data
X = df[['hours_to_expiry', 'current_stock', 'competitor_price']]
y = df['optimal_discount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the XGBoost Model
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Model Error Margin (RMSE): {rmse:.2f}%")

# 5. Export the Model (This is critical for real-time serving)
joblib.dump(model, 'pricing_engine.pkl')
print("Model compiled and saved as pricing_engine.pkl")