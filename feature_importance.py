# save as plot_feature_importance_xgb.py and run in project root
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from src import feature_extractor as fe  # uses prepare_features()

# paths
MODEL_PKL = "results/model.pkl"
CSV = "data/new_data_urls.csv"
OUT_PNG = "results/feature_importance_xgb.png"

# load data & features
df = pd.read_csv(CSV)
X_df = fe.prepare_features(df).drop(columns=["status"])

# try loading model
with open(MODEL_PKL, "rb") as f:
    model = pickle.load(f)

# get feature names in order used for training
feature_names = X_df.columns.tolist()

# if model is xgboost.Booster or sklearn wrapper
try:
    # sklearn wrapper (XGBClassifier)
    import xgboost as xgb
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif isinstance(model, xgb.core.Booster):
        # Booster: get score
        fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
        # get importance by gain (or 'weight','cover')
        imp_dict = model.get_score(importance_type="gain")
        # map back to array
        imp = [imp_dict.get(f"f{idx}", 0.0) for idx in range(len(feature_names))]
    else:
        raise ValueError("Model type not recognized for direct XGBoost importance.")
except Exception as e:
    raise RuntimeError("Failed to extract XGBoost importances: " + str(e))

# plot
imp_series = pd.Series(imp, index=feature_names).sort_values(ascending=True)
plt.figure(figsize=(8, max(4, 0.3 * len(imp_series))))
imp_series.plot(kind="barh")
plt.title("Feature importance (XGBoost)")
plt.xlabel("Importance (gain / feature_importances_)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print("Saved:", OUT_PNG)
plt.show()
