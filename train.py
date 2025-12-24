import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.feature_extractor import prepare_features
from src.model_utils import train_model, evaluate_model  # Updated import
import joblib

# Create results dir
os.makedirs('results', exist_ok=True)

# Load data
df = pd.read_csv('data/new_data_urls.csv')
print(f"Dataset shape: {df.shape}")
df.drop_duplicates(subset=['url'], inplace=True)
df.dropna( inplace=True)
print(f"Dataset shape after change: {df.shape}")
print(df['status'].value_counts(normalize=True))  # Check balance

# Feature extraction (this may take time on 800k+ rows; sample if needed)
X_df = prepare_features(df)
X = X_df.drop(columns=['status']).values
y = X_df['status'].values

# Train-test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train (now returns cv_f1_score)
model, best_params, cv_f1_score = train_model(X_train, y_train)
joblib.dump(model, 'results/model.pkl')

# Evaluate (pass cv_f1_score)
metrics = evaluate_model(model, X_test, y_test, cv_f1_score)

print(f"Best Params: {best_params}")
print("Model saved! Accuracy: {:.2%}".format(metrics['accuracy']))
# Expected: ~96% accuracy, F1 ~0.95 (based on balanced data and features)