import os
import urllib.request
from pathlib import Path

# YOLOv5 모델 다운로드 함수
def download_yolov5_model():
    model_url = "https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip"
    model_path = "yolov5.zip"
    extract_dir = "yolov5"

    # 모델 다운로드
    urllib.request.urlretrieve(model_url, model_path)

    # 압축 해제
    import zipfile
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 모델 경로 반환
    model_dir = Path(extract_dir) / "yolov5-master"
    return model_dir

# YOLOv5 모델 경로
model_dir = download_yolov5_model()
print("YOLOv5 모델 다운로드 완료:", model_dir)