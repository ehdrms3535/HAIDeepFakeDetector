#swin 4일차
cpu->gpu

no face crop-> face crop
train data 7만 -> 3만
SEED = 42

MODEL_ID = "microsoft/swin-base-patch4-window7-224"
IMG_SIZE = 224

TRAIN_NUM_FRAMES = 8
INFER_NUM_FRAMES = 16

AGG  mean-> topkmean      
TOPK_RATIO = 0.15     # topkmean 상위 비율(0.1~0.2 추천)
EPOCHS 16 -> 10
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0

1차 검증(1epo) 0.06/0.16/0.98 -> test score 0.34 이전보다 0.10 떨어짐
2차(3-4epo)
3차()
