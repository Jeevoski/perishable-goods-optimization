# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the real-time API
app = FastAPI(title="Dynamic Pricing Engine API")

# 2. Load the trained model into the server's memory
try:
    model = joblib.load('pricing_engine.pkl')
except FileNotFoundError:
    print("Error: Model file not found. Run train_model.py first.")

# 3. Define the strict data structure we expect from the vendor
class InventoryItem(BaseModel):
    hours_to_expiry: float
    current_stock: int
    competitor_price: float

# 4. Create the prediction endpoint
@app.post("/predict-discount")
def predict_discount(item: InventoryItem):
    # Convert incoming API data into a format the model understands
    input_data = pd.DataFrame([{
        'hours_to_expiry': item.hours_to_expiry,
        'current_stock': item.current_stock,
        'competitor_price': item.competitor_price
    }])
    
    # Generate the prediction in milliseconds
    prediction = model.predict(input_data)[0]
    
    # Return the business action
    return {
        "status": "success",
        "recommended_discount_percentage": round(float(prediction), 2),
        "action": f"Apply {round(float(prediction), 2)}% discount immediately to clear stock."
    }