# HAIDeepFakeDetector
2026 - HAI(하이)! - Hecto AI Challenge : 2025 하반기 헥토 채용 AI 경진대회

dataset
https://www.kaggle.com/datasets/yihaopuah/deep-fake-images

no crob dataset
[1/8] train_loss=0.6916  val_loss=0.6640  val_auc=0.6400
✅ saved best -> model\model.pt
Train:   0%|                                                                                          | 0/15 [00:00<?, ?it/s]C:\Users\Kim\HAI\your_submission\train.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
[2/8] train_loss=0.5299  val_loss=0.4959  val_auc=0.8533
✅ saved best -> model\model.pt
Train:   0%|                                                                                          | 0/15 [00:00<?, ?it/s]C:\Users\Kim\HAI\your_submission\train.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
[3/8] train_loss=0.2990  val_loss=0.7374  val_auc=0.8222
Train:   0%|                                                                                          | 0/15 [00:00<?, ?it/s]C:\Users\Kim\HAI\your_submission\train.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
[4/8] train_loss=0.1749  val_loss=0.6292  val_auc=0.8533
Train:   0%|                                                                                          | 0/15 [00:00<?, ?it/s]C:\Users\Kim\HAI\your_submission\train.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
[5/8] train_loss=0.1597  val_loss=0.7658  val_auc=0.8756
✅ saved best -> model\model.pt
Train:   0%|                                                                                          | 0/15 [00:00<?, ?it/s]C:\Users\Kim\HAI\your_submission\train.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
[6/8] train_loss=0.1403  val_loss=0.7830  val_auc=0.8711
Train:   0%|                                                                                          | 0/15 [00:00<?, ?it/s]C:\Users\Kim\HAI\your_submission\train.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
[7/8] train_loss=0.1204  val_loss=0.7916  val_auc=0.8756
Train:   0%|                                                                                          | 0/15 [00:00<?, ?it/s]C:\Users\Kim\HAI\your_submission\train.py:357: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
[8/8] train_loss=0.1171  val_loss=0.7955  val_auc=0.8756


crop
ch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
[1/8] train_loss=0.6794  val_loss=0.6542  val_auc=0.6978                                                                                  
✅ saved best -> model\model_facecrop.pt
[2/8] train_loss=0.4674  val_loss=0.5916  val_auc=0.7911
✅ saved best -> model\model_facecrop.pt
[3/8] train_loss=0.2650  val_loss=0.7862  val_auc=0.8267
✅ saved best -> model\model_facecrop.pt
[4/8] train_loss=0.1372  val_loss=0.7714  val_auc=0.8444
✅ saved best -> model\model_facecrop.pt
[5/8] train_loss=0.1232  val_loss=0.6917  val_auc=0.8533
✅ saved best -> model\model_facecrop.pt
[6/8] train_loss=0.0809  val_loss=0.7034  val_auc=0.8622
✅ saved best -> model\model_facecrop.pt
[7/8] train_loss=0.0768  val_loss=0.7503  val_auc=0.8578
[8/8] train_loss=0.0478  val_loss=0.7459  val_auc=0.8578
Done. Final weight: model\model_facecrop.pt