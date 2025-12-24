from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory

import joblib
import pandas as pd

from src.feature_extractor import extract_features


def create_app() -> Flask:
    # Serve the frontend from the web/ directory
    app = Flask(__name__, static_folder='web', static_url_path='')

    # Load model artifacts once at startup
    model_path = os.path.join('results', 'model.pkl')
    scaler_path = os.path.join('results', 'scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise RuntimeError('Model/scaler not found in results/. Please run training first.')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    threshold = 0.2  # Keep in sync with src.model_utils.predict_single

    def predict_url(url: str) -> Dict[str, Any]:
        feats = extract_features(url)
        feat_df = pd.DataFrame([feats])
        if 'status' in feat_df.columns:
            feat_df = feat_df.drop(columns=['status'])

        X_scaled = scaler.transform(feat_df.values)
        prob_legit = float(model.predict_proba(X_scaled)[0][1])

        # Hybrid rule as in src.model_utils.predict_single
        rule_match = (
            feats['url_length'] < 25 and
            feats['path_length'] == 0 and
            feats['num_sensitive'] == 0 and
            feats['has_https'] == 1
        )

        pred_legit = 1 if prob_legit >= threshold else 0
        if rule_match:
            pred_legit = 1
            prob_legit = max(prob_legit, 0.95)

        verdict = 'Legitimate' if pred_legit == 1 else 'Phishing'
        probability = prob_legit if pred_legit == 1 else 1.0 - prob_legit
        message = (
            'This URL looks safe based on current signals.'
            if verdict == 'Legitimate'
            else 'This URL exhibits suspicious characteristics.'
        )

        return {
            'verdict': verdict,
            'probability': probability,
            'message': message,
            'details': {
                'probability_legit': prob_legit,
                'threshold_used': threshold,
                'rule_applied': rule_match
            }
        }

    # Static index
    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    # Prediction endpoint
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(silent=True) or {}
        url = (data.get('url') or '').strip()
        if not url:
            return jsonify({'error': 'Missing url'}), 400
        try:
            result = predict_url(url)
            return jsonify(result)
        except Exception as exc:  # pragma: no cover - simple surface error
            return jsonify({'error': 'Failed to predict', 'detail': str(exc)}), 500

    return app


app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='127.0.0.1', port=port, debug=True)


