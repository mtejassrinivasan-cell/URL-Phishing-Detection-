import sys
from src.model_utils import predict_single

if len(sys.argv) != 2:
    print("Usage: python predict.py <url>")
    sys.exit(1)

url = sys.argv[1]
result = predict_single('results/model.pkl', 'results/scaler.pkl', url)

print(f"URL: {url}")
print(f"Prediction: {result['label']} (Class {result['prediction']})")
print(f"Legitimate Probability: {result['probability_legit']:.4f}")
print(f"Threshold Used: {result['threshold_used']}")