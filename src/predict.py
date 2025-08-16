# src/predict.py
import joblib, pandas as pd
from model_utils import recommend_next_crops_with_score

# Load saved objects
model = joblib.load("../models/crop_recommender.pkl")
label_encoders = joblib.load("../models/label_encoders.pkl")
original_df = joblib.load("../models/original_dataset.pkl")

# Example usage
results = recommend_next_crops_with_score("Wheat", original_df)

# Save predictions
pred_df = pd.DataFrame(results)
pred_df.to_csv("../output/predictions.csv", index=False)

print("✅ Predictions saved to output/predictions.csv")