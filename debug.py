import pandas as pd
import joblib
from src.feature_extractor import extract_features

# Load model/scaler (assumes trained)
model = joblib.load('results/model.pkl')
scaler = joblib.load('results/scaler.pkl')

url = "https://google.com"
feats = extract_features(url)
feat_df = pd.DataFrame([feats])

# Print features (for inspection)
print("Features for 'https://google.com':")
print(feat_df)

# Raw prediction details
if 'status' in feat_df.columns:
    feat_df = feat_df.drop(columns=['status'])
X_scaled = scaler.transform(feat_df.values)  # Fix: .values for no warning
raw_pred = model.predict(X_scaled)[0]
raw_proba = model.predict_proba(X_scaled)[0]
print(f"\nRaw Prediction: {raw_pred} (0=Phishing, 1=Legitimate)")
print(f"Probabilities: Phishing={raw_proba[0]:.4f}, Legitimate={raw_proba[1]:.4f}")