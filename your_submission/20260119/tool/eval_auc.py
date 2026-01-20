from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ====== 너 환경에 맞게 수정 ======
DATA_ROOT = Path(r"C:\Users\Kim\HAI\your_submission\train_data")
#FOLDS_CSV = "folds_with_pred_clean.csv"
FOLDS_CSV = "folds_with_pred_hard.csv"
PRED_COL = "pred"   # 너 모델이 만든 예측 확률 컬럼명 (0~1)
LOG_OUT = "auc_log.csv"
# ===============================
print("USING FOLDS_CSV =", FOLDS_CSV)
print("PRED_COL =", PRED_COL)

def abs_path(rel: str) -> str:
    rel = str(rel).strip().replace("\\", "/")
    return str(DATA_ROOT / rel)

def hard_transform_pil(img: Image.Image) -> Image.Image:
    """
    테스트 도메인 흉내(저품질 JPEG + 다운/업샘플).
    너무 과하면 역효과라 기본은 '적당히'로.
    """
    img = img.convert("RGB")

    # 1) 다운샘플 -> 업샘플
    w, h = img.size
    scale = 0.65  # 필요하면 0.5~0.8 사이 튜닝
    nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    # 2) JPEG 재압축
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=35)  # 필요하면 20~60 튜닝
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img

def main():
    df = pd.read_csv(FOLDS_CSV)
    assert {"path","label","fold"}.issubset(df.columns), "folds.csv 컬럼 확인 필요"

    # val만 평가
    val = df[df["fold"] == "val"].copy().reset_index(drop=True)
    if PRED_COL not in val.columns:
        print(f"주의: folds.csv에 {PRED_COL} 컬럼이 없음.")
        print("-> 너 모델 추론 결과(pred)를 folds.csv에 merge하거나, 별도 pred.csv를 merge해야 함.")
        return

    y = val["label"].astype(int).to_numpy()
    pred_clean = val[PRED_COL].astype(float).to_numpy()

    # Clean AUC
    auc_clean = roc_auc_score(y, pred_clean)

    # Hard AUC: 이미지에 hard 변환을 적용한 “재추론”이 원칙인데,
    # 여기서는 파이프라인 연결을 위해 'hard 이미지 저장 + 너 모델로 재추론'을 추천.
    # (아래는 hard 이미지 임시 저장만 생성)
    hard_dir = Path("./val_hard")
    hard_dir.mkdir(parents=True, exist_ok=True)

    abs_paths = val["path"].apply(abs_path).tolist()

    # hard 이미지 저장
    for rel, ap in tqdm(list(zip(val["path"].tolist(), abs_paths)), desc="make hard val images"):
        out_path = hard_dir / Path(rel).name
        if out_path.exists():
            continue
        try:
            img = Image.open(ap)
            himg = hard_transform_pil(img)
            himg.save(out_path, quality=95)  # 저장 품질은 크게 의미 없음(이미 변환됨)
        except Exception:
            pass

    print(f"[Clean AUC] {auc_clean:.6f}")
    print(f"hard val images saved to: {hard_dir.resolve()}")
    print("이제 너 추론 스크립트로 val_hard 폴더를 돌려서 hard 예측을 만든 뒤, 다시 AUC 계산해라.")

    # 로그 기록(클린만 먼저)
    row = pd.DataFrame([{
        "run": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "auc_clean": auc_clean,
        "note": "hard는 val_hard 재추론 후 merge 필요"
    }])
    if Path(LOG_OUT).exists():
        old = pd.read_csv(LOG_OUT)
        pd.concat([old, row], ignore_index=True).to_csv(LOG_OUT, index=False)
    else:
        row.to_csv(LOG_OUT, index=False)
    print("logged to:", LOG_OUT)

if __name__ == "__main__":
    main()
