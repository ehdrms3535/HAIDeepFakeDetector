start
C:\Users\Kim\HAI\.venv\Scripts\python.exe train.py train --face_crop --train_num_frames 8 --epochs 6 --batch_size 16 --agg topkmean --topk_ratio 0.15 --unfreeze_last_k 1 --remake_csv

resume
C:\Users\Kim\HAI\.venv\Scripts\python.exe train.py train --face_crop --train_num_frames 8 --epochs 10 --batch_size 16 --agg topkmean --topk_ratio 0.15 --unfreeze_last_k 1 --resume
