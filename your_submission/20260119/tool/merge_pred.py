import pandas as pd
from pathlib import Path

FOLDS_CSV = "folds.csv"
PRED_CSV  = "pred_clean.csv"
OUT_CSV   = "./tooloutput/folds_with_pred_clean.csv"

folds = pd.read_csv(FOLDS_CSV)
preds = pd.read_csv(PRED_CSV)

# folds: path -> filename 생성
if "path" not in folds.columns:
    raise KeyError(f"folds.csv에 'path' 컬럼이 없습니다. columns={list(folds.columns)}")

folds["filename"] = folds["path"].astype(str).apply(lambda x: Path(x).name)

# preds: filename 정규화(혹시 경로가 들어있어도 파일명만)
if "filename" not in preds.columns:
    raise KeyError(f"pred_clean.csv에 'filename' 컬럼이 없습니다. columns={list(preds.columns)}")

preds["filename"] = preds["filename"].astype(str).apply(lambda x: Path(x).name)

# merge
df = folds.merge(preds[["filename", "pred"]], on="filename", how="left")

missing_all = df["pred"].isna().sum()
missing_val = df[df["fold"].astype(str).str.lower() == "val"]["pred"].isna().sum()

print(f"missing pred (all): {missing_all}/{len(df)}")
print(f"missing pred (val): {missing_val}/{(df['fold'].astype(str).str.lower()=='val').sum()}")

# 혹시라도 mismatch 있으면 예시 출력
if missing_all > 0:
    miss = df[df["pred"].isna()][["path","filename","fold"]].head(20)
    print("missing examples:\n", miss)

assert missing_val == 0, "❌ val에 pred 누락이 있습니다. filename 매칭 규칙을 다시 맞춰야 함."

df.to_csv(OUT_CSV, index=False)
print("saved ->", OUT_CSV)
