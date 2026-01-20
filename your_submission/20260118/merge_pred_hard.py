import pandas as pd
from pathlib import Path

folds = pd.read_csv("folds.csv")
preds = pd.read_csv("pred_hard.csv")

folds["filename"] = folds["path"].astype(str).apply(lambda x: Path(x).name)
preds["filename"] = preds["filename"].astype(str).apply(lambda x: Path(x).name)

df = folds.merge(preds[["filename","pred"]], on="filename", how="left")

missing_val = df[df["fold"].astype(str).str.lower()=="val"]["pred"].isna().sum()
print("missing pred in val:", missing_val)

assert missing_val == 0, "val에 pred 누락(파일명 매칭 문제)"

df.to_csv("folds_with_pred_hard.csv", index=False)
print("saved -> folds_with_pred_hard.csv")
