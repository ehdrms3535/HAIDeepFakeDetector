"""
적응형 프레임 추론 실행 스크립트

사용법:
    python run_adaptive_inference.py

특징:
- 비디오 길이에 따라 프레임 수 자동 조정 (16/24/32)
- 이미지와 비디오 통합 처리
- 가중 집계 (평균 + 최대값 + 중앙값)
"""
import sys
from pathlib import Path
import torch

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from inference_adaptive import run_adaptive_inference


def main():
    """메인 실행 함수"""
    
    # ========================================
    # 설정 (필요시 수정)
    # ========================================
    
    # 모델 경로 (step1 모델 사용)
    MODEL_PATH = "./checkpoints/step1/best_model.pt"
    
    # 테스트 데이터 경로
    TEST_DIR = "./open/test_data"
    
    # 출력 CSV 경로
    OUTPUT_CSV = "./submissions/adaptive/submission.csv"
    
    # 모델 설정
    MODEL_NAME = "convnext_small"  # step1은 small 모델
    IMAGE_SIZE = 224
    USE_FACE_DETECTION = True
    
    # 디바이스 자동 선택
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========================================
    # 경로 확인
    # ========================================
    
    print("\n" + "="*60)
    print("적응형 프레임 추론 설정")
    print("="*60)
    print(f"모델 경로: {MODEL_PATH}")
    print(f"테스트 데이터: {TEST_DIR}")
    print(f"출력 CSV: {OUTPUT_CSV}")
    print(f"모델: {MODEL_NAME}")
    print(f"디바이스: {DEVICE}")
    print("="*60 + "\n")
    
    # 모델 파일 확인
    if not Path(MODEL_PATH).exists():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("\n학습이 완료되었는지 확인하세요.")
        print("Colab에서 학습한 경우, best_model.pt를 다운로드하여 해당 경로에 배치하세요.")
        return
    
    # 테스트 데이터 확인
    if not Path(TEST_DIR).exists():
        print(f"❌ 오류: 테스트 데이터 디렉토리를 찾을 수 없습니다: {TEST_DIR}")
        return
    
    # 출력 디렉토리 생성
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 추론 실행
    # ========================================
    
    try:
        df = run_adaptive_inference(
            model_path=MODEL_PATH,
            test_dir=TEST_DIR,
            output_csv=OUTPUT_CSV,
            model_name=MODEL_NAME,
            image_size=IMAGE_SIZE,
            use_face_detection=USE_FACE_DETECTION,
            device=DEVICE
        )
        
        print("\n" + "="*60)
        print("✓ 추론 성공!")
        print("="*60)
        print(f"제출 파일: {OUTPUT_CSV}")
        print(f"총 예측: {len(df)}개")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
