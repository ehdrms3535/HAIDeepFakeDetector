"""
제출 파일 생성 스크립트

사용법:
    ./venv/bin/python run_submission.py

경로 변경:
    config.py에서 다음 항목 수정
    - inference_model_path: 사용할 모델 경로
    - test_data_dir: 테스트 데이터 경로
    - submission_dir: 제출 파일 저장 경로
"""
from inference import inference
from config import Config
from pathlib import Path


def main():
    # 설정 로드
    config = Config()
    
    # 출력 디렉토리 생성
    Path(config.submission_dir).mkdir(parents=True, exist_ok=True)
    output_csv = Path(config.submission_dir) / "submission.csv"
    
    # 설정 출력
    print("=" * 60)
    print("제출 파일 생성")
    print("=" * 60)
    print(f"모델: {config.inference_model_path}")
    print(f"테스트 데이터: {config.test_data_dir}")
    print(f"출력 파일: {output_csv}")
    print(f"얼굴 검출: {config.face_detection}")
    print(f"비디오 프레임 수: {config.num_frames_per_video}")
    print(f"디바이스: {config.device}")
    print("=" * 60 + "\n")
    
    # 모델 파일 존재 확인
    if not Path(config.inference_model_path).exists():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {config.inference_model_path}")
        print("\n해결 방법:")
        print("1. config.py에서 inference_model_path 경로 확인")
        print("2. 모델이 학습되었는지 확인 (train.py 실행)")
        return
    
    # 테스트 데이터 존재 확인
    if not Path(config.test_data_dir).exists():
        print(f"❌ 오류: 테스트 데이터를 찾을 수 없습니다: {config.test_data_dir}")
        print("\n해결 방법:")
        print("1. config.py에서 test_data_dir 경로 확인")
        print("2. 테스트 데이터가 올바른 위치에 있는지 확인")
        return
    
    # 추론 실행
    try:
        df = inference(
            model_path=config.inference_model_path,
            test_dir=config.test_data_dir,
            output_csv=str(output_csv),
            model_name=config.model_name,
            image_size=config.image_size,
            use_face_detection=config.face_detection,
            num_frames=config.num_frames_per_video,
            device=config.device
        )
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("✓ 제출 파일 생성 완료!")
        print("=" * 60)
        print(f"출력 파일: {output_csv}")
        print(f"총 예측 수: {len(df)}")
        print("\n첫 10개 예측:")
        print(df.head(10).to_string(index=False))
        print("\n확률 통계:")
        print(df['prob'].describe())
        print("\n확률 분포:")
        print(f"  Real (< 0.5): {(df['prob'] < 0.5).sum()}개")
        print(f"  Fake (>= 0.5): {(df['prob'] >= 0.5).sum()}개")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
