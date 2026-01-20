import pandas as pd
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("pred_clean.csv")
auc = roc_auc_score(df["label"], df["pred"])
print("AUC:", auc)

folds = pd.read_csv("folds.csv")
pred  = pd.read_csv("pred_clean.csv")

# folds에서 val만
val = folds[folds["fold"]=="val"].copy()
val["filename"] = val["path"].apply(lambda x: Path(str(x)).name)

m = val.merge(pred[["filename"]], on="filename", how="left", indicator=True)
print("val total:", len(val))
print("pred matched:", (m["_merge"]=="both").sum())
print("missing:", (m["_merge"]!="both").sum())
print("missing examples:", m[m["_merge"]!="both"]["filename"].head(20).tolist())
