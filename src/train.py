# src/train.py
import pandas as pd
import joblib, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import encode_datasets

# Load datasets
original_df = pd.read_excel("../data/Originaldata.csv")
pairwise_df = pd.read_excel("../data/pairwise.csv")

# Encode
pairwise_encoded, label_encoders = encode_datasets(original_df, pairwise_df)

# Features & target
X = pairwise_encoded[['Current Crop', 'Next Crop', 'Soil Type', 'Current Season', 'Next Season']]
y = pairwise_encoded['Pest Overlap']

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_df = pd.concat([X_val, y_val], axis=1)
val_df.to_csv("../data/validation.csv", index=False)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred, output_dict=True)

print("Validation Accuracy:", acc)

# Save model
joblib.dump(model, "../models/crop_recommender.pkl")
joblib.dump(label_encoders, "../models/label_encoders.pkl")
joblib.dump(original_df, "../models/original_dataset.pkl")

# Save model architecture JSON
model_arch = {"model": "RandomForestClassifier", "n_estimators": 100, "random_state": 42}
with open("../models/model_architecture.json", "w") as f:
    json.dump(model_arch, f, indent=4)

# Save metrics
with open("../output/metrics.json", "w") as f:
    json.dump({"accuracy": acc, "classification_report": report}, f, indent=4)

with open("../output/training_log.txt", "w") as f:
    f.write("Validation Accuracy: {}\n".format(acc))
    f.write(str(report))

print("✅ Training complete, model & metrics saved!")