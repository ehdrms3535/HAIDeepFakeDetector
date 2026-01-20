import pandas as pd
from sklearn.metrics import roc_auc_score

FOLDS_CSV = "folds.csv"

# 네 추론 결과 CSV (파일명 기준으로 맞춘다고 가정)
# 형식 예: filename,pred  (filename은 real_123.jpg 같은 베이스네임)
PRED_CLEAN = "pred_clean.csv"
PRED_HARD  = "pred_hard.csv"

def main():
    df = pd.read_csv(FOLDS_CSV)
    val = df[df["fold"] == "val"].copy()

    # filename 컬럼 만들기
    val["filename"] = val["path"].apply(lambda p: str(p).replace("\\","/").split("/")[-1])

    pc = pd.read_csv(PRED_CLEAN)
    ph = pd.read_csv(PRED_HARD)

    # pred 컬럼명 통일
    pc = pc.rename(columns={pc.columns[0]: "filename", pc.columns[1]: "pred_clean"})
    ph = ph.rename(columns={ph.columns[0]: "filename", ph.columns[1]: "pred_hard"})

    val = val.merge(pc, on="filename", how="left")
    val = val.merge(ph, on="filename", how="left")

    val = val.dropna(subset=["pred_clean","pred_hard"])

    y = val["label"].astype(int).to_numpy()
    auc_clean = roc_auc_score(y, val["pred_clean"].astype(float).to_numpy())
    auc_hard  = roc_auc_score(y, val["pred_hard"].astype(float).to_numpy())

    print(f"AUC clean = {auc_clean:.6f}")
    print(f"AUC hard  = {auc_hard:.6f}")

if __name__ == "__main__":
    main()
