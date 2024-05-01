import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from yolov5.yolov5_master.models import yolo
import yaml

# YOLOv5 모델 초기화
model = yolo()

# 데이터 로드 함수 정의
def load_data(data_config):
    with open(data_config) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data['train']
    val_path = data['val']
    return train_path, val_path

# 주어진 데이터셋의 요구사항 확인
def check_requirements():
    if not os.path.exists('yolov5'):
        print("Downloading YOLOv5 repository...")
        os.system("git clone https://github.com/ultralytics/yolov5")

# 하이퍼파라미터 및 경로 설정
data_config = 'data.yaml'
epochs = 30
batch_size = 16

# 데이터 로드
train_path, val_path = load_data(data_config)

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((416, 416)),  # YOLOv5 모델의 입력 크기에 맞게 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델 학습
for epoch in range(epochs):
    model.train()
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(model.device)
        labels = labels.to(model.device)
        
        # Forward pass
        outputs = model(imgs)
        
        # Loss 계산
        loss = model.compute_loss(outputs, labels)
        
        # Backward pass 및 가중치 업데이트
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

# 학습된 모델 저장
torch.save(model.state_dict(), 'trained_model.pt')