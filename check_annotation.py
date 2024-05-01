import cv2

# 텍스트 파일 경로 설정
annotation_path = "D:/dataset/labels/train/train_txt/s01000200.txt"

# 이미지 파일 경로 설정
image_path = "D:/dataset/images/train/image1/s01000200.jpg"

# 이미지 로드
image = cv2.imread(image_path)

# 텍스트 파일 읽기
with open(annotation_path, 'r') as f:
    lines = f.readlines()

# bounding box 정보 추출하여 이미지에 그리기
for line in lines:
    # 줄을 공백으로 나누어서 각 요소 추출
    parts = line.strip().split()
    if len(parts) == 5:  # 각 요소의 개수가 5개인 경우에만 처리
        xmin, ymin, xmax, ymax = map(int, parts[1:])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# 이미지 출력
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()