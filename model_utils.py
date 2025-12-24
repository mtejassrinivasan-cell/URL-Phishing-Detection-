import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV  # Changed to Randomized
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  # For predict_single
from typing import Tuple, Dict
from scipy.stats import uniform, randint  # For RandomizedSearch distributions

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[xgb.XGBClassifier, dict, float]:
    """
    Trains XGBoost with randomized hyperparameter tuning and CV.
    Returns: model, best_params, cv_f1_score
    """
    # Scale features (helps XGBoost stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # XGBoost base params
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'base_score': 0.5,  # Balanced logistic
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Distributions for RandomizedSearchCV (covers same space as grid)
    param_dist = {
        'n_estimators': randint(100, 200),  # 100-200
        'max_depth': randint(3, 8),  # 3-7
        'learning_rate': uniform(0.05, 0.20),  # 0.05-0.20
        'subsample': uniform(0.8, 1.0),  # 0.8-1.0
        'colsample_bytree': uniform(0.8, 1.0),  # 0.8-1.0
        'gamma': uniform(0,  0.2),  # 0-0.2
        'reg_lambda': uniform(1.0,  2.0)  # 1.0-2.0
    }
    
    model = xgb.XGBClassifier(**base_params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # RandomizedSearchCV: n_iter=50 random fits (vs 288 full)
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=100,  # Reduce here; increase for more precision
        cv=cv, scoring='f1', n_jobs=-1, verbose=1, random_state=42
    )
    random_search.fit(X_train_scaled, y_train)
    
    # Best model with early stopping
    best_params = random_search.best_params_
    best_params.update(base_params)
    best_params['early_stopping_rounds'] = 10
    best_params['scale_pos_weight'] = 1.0  # Balanced
    
    best_model = xgb.XGBClassifier(**best_params)
    
    best_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train)],
        verbose=False
    )
    
    # Save scaler
    joblib.dump(scaler, 'results/scaler.pkl')
    
    cv_f1_score = random_search.best_score_
    return best_model, random_search.best_params_, cv_f1_score

def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray, cv_f1_score: float, scaler_path: str = 'results/scaler.pkl') -> Dict:
    """
    Evaluates model with metrics. Returns dict for JSON save.
    """
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Feature importance plot
    xgb.plot_importance(model)
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics.update(report['1'])  # For class 1 (legit)
    
    # Save metrics
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Cross-Validation F1: {cv_f1_score:.4f}")
    print("Test Metrics:", metrics)
    
    return metrics

def predict_single(model_path: str, scaler_path: str, url: str) -> Dict[str, any]:
    """
    Predicts on a single URL with hybrid ML + rule for common legit patterns.
    """
    from src.feature_extractor import extract_features
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    feats = extract_features(url)
    feat_df = pd.DataFrame([feats])
    
    if 'status' in feat_df.columns:
        feat_df = feat_df.drop(columns=['status'])
    
    X_scaled = scaler.transform(feat_df.values)
    
    prob = model.predict_proba(X_scaled)[0][1]  # Raw ML prob for legit
    
    # ML threshold (0.2 tuned for balance)
    threshold = 0.2
    pred = 1 if prob >= threshold else 0
    
    # Hybrid Rule: Boost for common legit homepages (short, root, HTTPS, no sensitive)
    rule_match = (feats['url_length'] < 25 and 
                  feats['path_length'] == 0 and 
                  feats['num_sensitive'] == 0 and 
                  feats['has_https'] == 1)
    if rule_match:
        pred = 1
        prob = max(prob, 0.95)
    
    label = 'Legitimate' if pred == 1 else 'Phishing'
    
    return {'prediction': int(pred), 'probability_legit': float(prob), 'label': label, 'threshold_used': threshold, 'rule_applied': rule_match}