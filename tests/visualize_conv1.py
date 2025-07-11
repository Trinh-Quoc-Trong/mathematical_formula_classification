import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path để có thể import từ src [[memory:2479219]]
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import kiến trúc model từ file src/models/model.py [[memory:2479219]]
from src.models.model import SimpleCNN

def visualize_first_layer_output(model, image_path, transform):
    """
    Tải một ảnh, cho nó đi qua lớp conv đầu tiên của model,
    và vẽ các feature map đầu ra.
    """
    # --- 1. Tải và tiền xử lý ảnh ---
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại: {image_path}")
        return

    # Áp dụng các phép biến đổi
    # unsqueeze(0) để thêm chiều batch (từ [C, H, W] thành [B, C, H, W], ở đây B=1)
    # Model mong đợi input là một batch, dù chỉ có 1 ảnh.
    image_tensor = transform(image).unsqueeze(0)
    
    print(f"Kích thước ảnh đầu vào sau khi xử lý: {image_tensor.shape}")

    # --- 2. Đưa ảnh qua các lớp ---
    # Đặt model ở chế độ evaluation
    model.eval()

    # Không cần tính gradient cho việc này
    with torch.no_grad():
        # Bước 1: Cho ảnh đi qua lớp conv đầu tiên
        conv1_output = model.conv1(image_tensor)
        print(f"Kích thước đầu ra của conv1: {conv1_output.shape}")

        # Bước 2: Áp dụng hàm kích hoạt ReLU
        relu_output = F.relu(conv1_output)
        print(f"Kích thước sau khi qua ReLU: {relu_output.shape}")

        # Bước 3: Cho qua lớp pooling đầu tiên
        pool1_output = model.pool1(relu_output)
        print(f"Kích thước sau khi qua pool1: {pool1_output.shape}")

    # --- 3. Trực quan hóa các feature map từ lớp conv1 ---
    # Lấy tensor đầu ra của conv1, loại bỏ chiều batch
    feature_maps = conv1_output.squeeze(0)

    # Tạo một figure để vẽ
    # Chúng ta có 16 bộ lọc (out_channels=16), nên sẽ có 16 feature map
    # Ta sẽ vẽ chúng trên một lưới 4x4
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Đầu ra từ 16 bộ lọc của lớp Conv1', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[0]: # feature_maps.shape[0] là 16
            # Lấy ra feature map thứ i
            feature_map = feature_maps[i]
            
            # Vẽ feature map
            ax.imshow(feature_map.numpy(), cmap='gray')
            ax.set_title(f'Bộ lọc (Filter) #{i+1}')
        ax.axis('off') # Tắt các trục tọa độ

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Đường dẫn đến ảnh bạn muốn test
    IMAGE_PATH = r"data/processed/train/ky_hieu_tich/nhan_0_1_aug_7.png"

    # --- Khởi tạo model và các phép biến đổi ---
    # Không cần tải trọng số đã huấn luyện, vì ta chỉ muốn xem
    # cách bộ lọc (dù là ngẫu nhiên) biến đổi ảnh.
    cnn_model = SimpleCNN(num_classes=3)
    
    # Các phép biến đổi phải GIỐNG HỆT như trong file train_model.py
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    visualize_first_layer_output(model=cnn_model, image_path=IMAGE_PATH, transform=data_transforms) 