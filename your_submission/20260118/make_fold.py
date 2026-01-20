from pathlib import Path
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# ====== 너 환경에 맞게 수정 ======
DATA_ROOT = Path(r"C:\Users\Kim\HAI\your_submission\train_data")
CSV_IN = "train.csv"          # 전체 학습 csv (path,label)
CSV_OUT = "folds.csv"         # 결과: path,label,phash,group_id,fold
SEED = 42
VAL_RATIO = 0.2
# ===============================

def abs_path(rel: str) -> str:
    rel = str(rel).strip().replace("\\", "/")
    return str(DATA_ROOT / rel)

def phash_of_file(fp: str) -> str:
    img = Image.open(fp).convert("RGB")
    return str(imagehash.phash(img))

def main():
    df = pd.read_csv(CSV_IN)

    assert "path" in df.columns, "train.csv에 path 컬럼이 필요함"
    # label 컬럼명이 다르면 여기만 바꿔
    label_col = "label" if "label" in df.columns else df.columns[-1]

    # 절대경로로 존재 확인
    df["abs_path"] = df["path"].apply(abs_path)
    ok = df["abs_path"].apply(lambda p: Path(p).exists()).mean()
    print(f"[exists rate] {ok:.2%}")
    if ok < 0.99:
        print("DATA_ROOT 또는 path가 맞는지 확인 필요")
        # 계속 진행은 하되, phash가 많이 실패할 수 있음

    # pHash 계산
    phashes = []
    fail = 0
    for p in tqdm(df["abs_path"].tolist(), desc="compute pHash"):
        try:
            phashes.append(phash_of_file(p))
        except Exception:
            phashes.append(None)
            fail += 1
    df["phash"] = phashes
    print(f"[phash] fail={fail}/{len(df)}")

    # phash 실패는 제거(권장)
    df = df[df["phash"].notna()].reset_index(drop=True)

    # 그룹: 동일 phash는 같은 그룹
    # group_id는 phash 자체를 써도 됨
    df["group_id"] = df["phash"]

    # 그룹 단위 split
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    (train_idx, val_idx) = next(gss.split(df, groups=df["group_id"]))

    df["fold"] = "train"
    df.loc[val_idx, "fold"] = "val"

    # 출력
    out = df[["path", label_col, "phash", "group_id", "fold"]].rename(columns={label_col: "label"})
    out.to_csv(CSV_OUT, index=False)
    print("saved:", CSV_OUT)
    print(out["fold"].value_counts())

if __name__ == "__main__":
    main()
