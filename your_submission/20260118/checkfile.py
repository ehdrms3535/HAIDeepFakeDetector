from pathlib import Path

import re
from pathlib import Path
from collections import Counter
from PIL import Image
import imagehash
from tqdm import tqdm
import random
import pandas as pd

def normalize_name(p: str) -> str:
    # 경로가 들어와도 파일명만 비교
    return Path(p).name.strip()

def overlap_by_filename(train_list, val_list):
    train_set = set(map(normalize_name, train_list))
    val_set   = set(map(normalize_name, val_list))
    inter = train_set & val_set
    print(f"[filename overlap] train={len(train_set)}, val={len(val_set)}, overlap={len(inter)}")
    if inter:
        print("examples:", list(sorted(inter))[:20])
    return inter


def extract_video_id(name: str) -> str:
    """
    filename에서 video_id 추출.
    기본 전략:
    - 'TEST_000' / 'TRAIN_123' 같은 패턴 우선
    - 그게 없으면 '_'로 쪼개서 앞 2토큰 사용 (예: TEST_000_frame10 -> TEST_000)
    필요하면 너 파일명 규칙에 맞춰 여기만 고치면 됨.
    """
    base = Path(name).stem  # 확장자 제거
    m = re.match(r'((?:TEST|TRAIN|VAL)_[0-9]+)', base, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0].upper()}_{parts[1]}"
    return parts[0].upper()

def overlap_by_video_id(train_list, val_list):
    train_ids = set(extract_video_id(p) for p in train_list)
    val_ids   = set(extract_video_id(p) for p in val_list)
    inter = train_ids & val_ids
    print(f"[video_id overlap] train_ids={len(train_ids)}, val_ids={len(val_ids)}, overlap={len(inter)}")
    if inter:
        print("examples:", list(sorted(inter))[:20])
    return inter



def frames_per_video_report(file_list, title="train"):
    ids = [extract_video_id(p) for p in file_list]
    c = Counter(ids)
    counts = sorted(c.values(), reverse=True)
    print(f"[{title}] unique_videos={len(c)}, total_frames={len(file_list)}")
    if counts:
        print(f"[{title}] frames/video: max={counts[0]}, median={counts[len(counts)//2]}, min={counts[-1]}")
    # 상위 몇개 보기
    top = sorted(c.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"[{title}] top videos:", top)
    return c



def compute_hashes(file_list, hash_type="phash", max_n=None):
    hfunc = {"phash": imagehash.phash, "ahash": imagehash.average_hash}[hash_type]
    hashes = {}
    it = file_list if max_n is None else file_list[:max_n]
    for p in tqdm(it, desc=f"hashing({hash_type})"):
        try:
            img = Image.open(p).convert("RGB")
            h = str(hfunc(img))
            hashes[p] = h
        except Exception:
            continue
    return hashes

def exact_duplicate_rate(train_files, val_files, hash_type="phash", train_max=50000):
    train_hash = compute_hashes(train_files, hash_type=hash_type, max_n=train_max)
    val_hash   = compute_hashes(val_files,   hash_type=hash_type, max_n=None)

    train_set = set(train_hash.values())
    dup = [p for p,h in val_hash.items() if h in train_set]
    rate = len(dup) / max(1, len(val_hash))
    print(f"[exact dup via {hash_type}] val={len(val_hash)} train_sample={len(train_hash)} dup={len(dup)} rate={rate:.4f}")
    if dup:
        print("examples:", dup[:10])
    return dup, rate


def near_duplicate_check(train_files, val_files, hash_type="phash",
                         train_max=8000, val_max=800, dist_th=5, seed=42):
    random.seed(seed)

    # 샘플링
    train_s = train_files if len(train_files) <= train_max else random.sample(train_files, train_max)
    val_s   = val_files   if len(val_files)   <= val_max   else random.sample(val_files,   val_max)

    hfunc = {"phash": imagehash.phash, "ahash": imagehash.average_hash}[hash_type]

    # train 해시 사전
    train_h = []
    for p in tqdm(train_s, desc=f"hash train({hash_type})"):
        try:
            img = Image.open(p).convert("RGB")
            train_h.append((p, hfunc(img)))
        except Exception:
            continue

    # val 각 이미지에 대해 가까운 train 찾기
    hits = []
    for vp in tqdm(val_s, desc="scan val vs train"):
        try:
            vimg = Image.open(vp).convert("RGB")
            vh = hfunc(vimg)
        except Exception:
            continue

        best = (10**9, None)
        for tp, th in train_h:
            d = vh - th  # 해시 거리
            if d < best[0]:
                best = (d, tp)
                if d <= dist_th:
                    break

        if best[0] <= dist_th:
            hits.append((vp, best[1], best[0]))

    rate = len(hits) / max(1, len(val_s))
    print(f"[near-dup {hash_type}] dist_th={dist_th} val_sample={len(val_s)} train_sample={len(train_h)} hits={len(hits)} rate={rate:.4f}")
    if hits:
        print("examples (val, train, dist):", hits[:10])
    return hits, rate



def save_hit_pairs(hits, out_dir="./dup_examples", max_pairs=20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, (vp, tp, d) in enumerate(hits[:max_pairs]):
        try:
            vimg = Image.open(vp).convert("RGB")
            timg = Image.open(tp).convert("RGB")
            vimg.save(out / f"{i:02d}_VAL_d{d}_{Path(vp).name}")
            timg.save(out / f"{i:02d}_TRAIN_d{d}_{Path(tp).name}")
        except Exception:
            pass
    print(f"saved {min(len(hits), max_pairs)} pairs to {out.resolve()}")


train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")

DATA_ROOT = Path(r"C:\Users\Kim\HAI\your_submission\train_data")

def resolve_path(p):
    p = str(p).strip().replace("\\", "/")
    pp = DATA_ROOT / Path(p)
    return str(pp)


train_files = [resolve_path(p) for p in train_df['path'].tolist()]
val_files   = [resolve_path(p) for p in val_df['path'].tolist()]

print(f"train_files={len(train_files)}  val_files={len(val_files)}")

print("=== path exists check ===")
for p in train_files[:5] + val_files[:5]:
    print(p, "=>", Path(p).exists())


def count_exists(paths):
    ok = sum(Path(p).exists() for p in paths)
    print(f"exists: {ok}/{len(paths)} ({ok/len(paths):.2%})")

count_exists(train_files)
count_exists(val_files)

# 1) 파일명 교집합
overlap_by_filename(train_files, val_files)

# 2) video_id 교집합 + 프레임 분포
overlap_by_video_id(train_files, val_files)
frames_per_video_report(train_files, "train")
frames_per_video_report(val_files, "val")

# 3-A) 완전 동일(해시 동일) 중복 체크 (빠르고 강력)
dup, dup_rate = exact_duplicate_rate(
    train_files, val_files,
    hash_type="phash",
    train_max=30000  # 5만이 오래 걸리면 3만이면 충분
)

# 3-B) 거의 동일(거리 기준) 중복 체크 (샘플로만)
hits, rate = near_duplicate_check(
    train_files, val_files,
    hash_type="phash",
    train_max=8000,
    val_max=800,
    dist_th=5
)
save_hit_pairs(hits, out_dir="./dup_examples", max_pairs=20)


