import csv
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("runs_log.csv")

# ===== 이번 run 정보만 여기 채우면 됨 =====
run_info = {
    "run_id": "swin_base_run1",      # 네가 정하는 이름
    "clean_auc": 0.9924011758608744,
    "hard_auc": 0.9760201634881025,
    "lb_score": 0.3348930481,                # ← 리더보드 점수 넣기 (예: 0.81234)
    "notes": "frames=16 infer, agg=topkmean0.15, face_crop, unfreeze_last_k=1",
}
# =========================================

# 타임스탬프 자동
run_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# CSV 헤더 순서 고정
fields = ["timestamp", "run_id", "clean_auc", "hard_auc", "lb_score", "notes"]

# 파일 없으면 헤더부터 생성
write_header = not LOG_PATH.exists()

with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()
    writer.writerow(run_info)

print("✅ appended to", LOG_PATH.resolve())
