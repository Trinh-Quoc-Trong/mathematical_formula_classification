"""Script ví dụ để chạy visualize inkml"""

from pathlib import Path
from test import visualize_inkml

# Danh sách các file mẫu
sample_files = [
    "handwritten-mathematical-expressions/versions/1/MatricesTest2014/MatricesTest/RIT_MatrixTest_2014_2.inkml",
    "handwritten-mathematical-expressions/versions/1/MatricesTest2014/MatricesTest/RIT_MatrixTest_2014_10.inkml",
    "handwritten-mathematical-expressions/versions/1/CROHME_test_2011/algb02.inkml",
]

# Hiển thị file đầu tiên tìm thấy
for file_path in sample_files:
    path = Path(file_path)
    if path.exists():
        print(f"Đang hiển thị: {file_path}")
        visualize_inkml(path)
        break
    else:
        print(f"Không tìm thấy: {file_path}")
else:
    print("Không tìm thấy file nào!") 