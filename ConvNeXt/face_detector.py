"""
얼굴 검출기
"""
import cv2
import numpy as np


class FaceDetector:
    """OpenCV 기반 얼굴 검출"""
    
    def __init__(self):
        # Haar Cascade 로드
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_face(self, image, margin=0.2):
        """
        이미지에서 얼굴 검출 및 크롭
        
        Args:
            image: BGR 이미지
            margin: 얼굴 주변 여백 비율
        
        Returns:
            크롭된 얼굴 이미지 또는 None
        """
        if image is None or image.size == 0:
            return None
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴 선택
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # 여백 추가
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        # 얼굴 크롭
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop if face_crop.size > 0 else None
    
    def crop_face_with_fallback(self, image, target_size=None):
        """
        얼굴 검출 시도, 실패 시 원본 반환
        
        Args:
            image: BGR 이미지
            target_size: 리사이즈 목표 크기 (width, height)
        
        Returns:
            크롭된 얼굴 이미지 또는 원본 이미지 (target_size로 리사이즈됨)
        """
        face = self.detect_face(image)
        result = face if face is not None else image
        
        # target_size가 지정된 경우 리사이즈
        if target_size is not None and result is not None:
            result = cv2.resize(result, target_size, interpolation=cv2.INTER_LINEAR)
        
        return result
