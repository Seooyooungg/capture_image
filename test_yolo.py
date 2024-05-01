import torch
from PIL import Image
import matplotlib.pyplot as plt
from yolov5.yolov5_master.models import yolo
from torchvision import transforms
import numpy as np

# 학습된 모델 불러오기
model = yolo()
model.load_state_dict(torch.load('trained_model.pt'))
model.eval()

# 테스트할 이미지 경로
test_image_path = 'D:/dataset/images/val/image1/s01000100.jpg'

# 이미지를 텐서로 변환
image = Image.open(test_image_path)
transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# 모델로부터 예측 얻기
with torch.no_grad():
    outputs = model(image_tensor)

# 예측 결과 시각화
def plot_image_with_boxes(image, boxes):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        xy = (box[0], box[1])
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = plt.Rectangle(xy, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()

# 예측된 바운딩 박스 추출
pred_boxes = outputs[0]['pred_boxes'].cpu().numpy()
pred_scores = outputs[0]['scores'].cpu().numpy()
pred_classes = outputs[0]['pred_classes'].cpu().numpy()

# 예측 결과 시각화
plot_image_with_boxes(np.array(image), pred_boxes)