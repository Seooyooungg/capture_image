import os
import json

# JSON 파일이 있는 디렉토리 경로
json_dir = "/path/to/json/files"

# 텍스트 파일을 저장할 디렉토리 경로 (D 드라이브)
output_dir = "D:/output"

# JSON 파일을 읽어서 텍스트 파일로 변환하여 저장하는 함수
def json_to_txt(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON 데이터를 텍스트로 변환
    text_data = json.dumps(data)
    
    # 텍스트 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text_data)

# JSON 파일을 읽어서 텍스트 파일로 변환하여 저장
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_file = os.path.join(json_dir, filename)
        output_file = os.path.join(output_dir, filename.replace(".json", ".txt"))
        json_to_txt(json_file, output_file)
