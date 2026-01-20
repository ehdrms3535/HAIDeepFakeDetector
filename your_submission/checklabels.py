import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv"}

def read_one_frame(path, max_side=960):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXT:
        img = cv2.imread(path)
        if img is None:
            return None
        return img

    if ext in VID_EXT:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = n // 2 if n > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        img = frame
        return img

    return None

def resize_keep(img, max_side=960):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / s
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def jpeg_like_score(img_bgr):
    """
    진짜 JPEG quality를 추정하긴 어렵고,
    '블록/압축 흔적'에 민감한 간단 지표(대략용)만 뽑음.
    """
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
    # 8x8 블록 경계에서의 평균 절댓값 차이
    v = np.mean(np.abs(y[:, 8:] - y[:, :-8])) + np.mean(np.abs(y[8:, :] - y[:-8, :]))
    return float(v)

def compute_row(path):
    img = read_one_frame(path)
    if img is None:
        return None
    img = resize_keep(img, 960)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = float(gray.mean())
    std = float(gray.std())
    sharp = lap_var(gray)
    comp = jpeg_like_score(img)
    return dict(w=w, h=h, mean=mean, std=std, sharp=sharp, comp=comp)

def main(csv_path="labels.csv"):
    df = pd.read_csv(csv_path)
    rows = []
    for p, y in tqdm(df[["path","label"]].values, total=len(df)):
        r = compute_row(p)
        if r is None:
            continue
        r["path"] = p
        r["label"] = int(y)
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv("style_stats.csv", index=False)

    # label별 요약
    g = out.groupby("label")[["w","h","mean","std","sharp","comp"]].agg(["mean","median","std"])
    print(g)

    # 간단한 경고(차이가 크면 편향 의심)
    real = out[out.label==0]
    fake = out[out.label==1]
    def gap(col):
        return float(fake[col].median() - real[col].median())

    print("\n[median gap] (fake - real)")
    for c in ["w","h","mean","std","sharp","comp"]:
        print(c, gap(c))

if __name__ == "__main__":
    main("labels.csv")
