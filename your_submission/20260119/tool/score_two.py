import pandas as pd
from sklearn.metrics import roc_auc_score

def auc_from_folds(csv_path, pred_col="pred"):
    df = pd.read_csv(csv_path)
    v = df[df["fold"].astype(str).str.lower() == "val"].copy()
    v = v.dropna(subset=[pred_col])
    return roc_auc_score(v["label"], v[pred_col])

print("Clean AUC:", auc_from_folds("folds_with_pred_clean.csv"))
print("Hard  AUC:", auc_from_folds("folds_with_pred_hard.csv"))
