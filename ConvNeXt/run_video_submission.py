"""
비디오 전용 제출 파일 생성

MP4 파일만 처리하여 높은 정확도 확보
"""
from inference_video import inference_videos
from config import Config
from pathlib import Path


def main():
    config = Config()
    
    # ========================================
    # 프레임 수 설정 (여기서 변경)
    # ========================================
    NUM_FRAMES = 32  # ← 32프레임으로 설정
    
    # 출력 디렉토리
    output_dir = Path(config.submission_dir) / f"video_{NUM_FRAMES}frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "submission_video.csv"
    
    print("=" * 60)
    print("비디오 전용 제출 파일 생성")
    print("=" * 60)
    print(f"모델: {config.inference_model_path}")
    print(f"테스트 데이터: {config.test_data_dir}")
    print(f"프레임 수: {NUM_FRAMES}개/비디오")  # ← 표시
    print(f"출력 파일: {output_csv}")
    print("=" * 60)
    
    # 모델 확인
    if not Path(config.inference_model_path).exists():
        print(f"❌ 모델 파일 없음: {config.inference_model_path}")
        return
    
    # 추론 실행
    df = inference_videos(
        model_path=config.inference_model_path,
        test_dir=config.test_data_dir,
        output_csv=str(output_csv),
        model_name=config.model_name,
        image_size=config.image_size,
        use_face_detection=config.face_detection,
        num_frames=NUM_FRAMES,  # ← 변수 사용
        device=config.device
    )
    
    # 결과 분석
    print("\n" + "=" * 60)
    print("✓ 비디오 추론 완료!")
    print("=" * 60)
    print(f"출력: {output_csv}")
    print(f"총 비디오: {len(df)}개")
    
    print("\n첫 10개 결과:")
    print(df.head(10).to_string(index=False))
    
    print("\n확률 통계:")
    stats = df['prob'].describe()
    print(stats)
    
    print("\n확률 분포:")
    print(f"  평균: {df['prob'].mean():.4f}")
    print(f"  중앙값: {df['prob'].median():.4f}")
    print(f"  Real (< 0.5): {(df['prob'] < 0.5).sum()}개 ({(df['prob'] < 0.5).sum()/len(df)*100:.1f}%)")
    print(f"  Fake (>= 0.5): {(df['prob'] >= 0.5).sum()}개 ({(df['prob'] >= 0.5).sum()/len(df)*100:.1f}%)")
    
    print("\n확률 범위별 분포:")
    print(f"  0.0-0.2: {((df['prob'] >= 0.0) & (df['prob'] < 0.2)).sum()}개")
    print(f"  0.2-0.4: {((df['prob'] >= 0.2) & (df['prob'] < 0.4)).sum()}개")
    print(f"  0.4-0.6: {((df['prob'] >= 0.4) & (df['prob'] < 0.6)).sum()}개")
    print(f"  0.6-0.8: {((df['prob'] >= 0.6) & (df['prob'] < 0.8)).sum()}개")
    print(f"  0.8-1.0: {((df['prob'] >= 0.8) & (df['prob'] <= 1.0)).sum()}개")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
