import os
import csv

REAL_DIR = "train_data/real"
FAKE_DIR = "train_data/fake"
OUT_CSV  = "labels.csv"

IMG_EXT = {".jpg",".jpeg",".png",".jfif",".webp"}
VID_EXT = {".mp4",".mov",".avi",".mkv"}
EXTS = IMG_EXT | VID_EXT

rows = []

def collect(dir_path, label):
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                rows.append([os.path.join(root, f), label])

collect(REAL_DIR, 0)
collect(FAKE_DIR, 1)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path","label"])
    w.writerows(rows)

print(f"saved {OUT_CSV}, total={len(rows)}")
