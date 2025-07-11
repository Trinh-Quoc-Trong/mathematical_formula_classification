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

def visualize_second_layer_output(model, image_path, transform):
    """
    Tải một ảnh, cho nó đi qua lớp conv1, pool1, rồi conv2,
    và vẽ các feature map đầu ra của conv2.
    """
    # --- 1. Tải và tiền xử lý ảnh ---
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại: {image_path}")
        return

    # Áp dụng các phép biến đổi và thêm chiều batch
    image_tensor = transform(image).unsqueeze(0)
    print(f"Kích thước ảnh đầu vào sau khi xử lý: {image_tensor.shape}")

    # --- 2. Đưa ảnh qua các lớp ---
    model.eval()

    with torch.no_grad():
        # Luồng dữ liệu để đến được conv2
        # Qua khối đầu tiên (conv1 -> relu -> pool1)
        x = model.pool1(F.relu(model.conv1(image_tensor)))
        print(f"Kích thước đầu vào cho conv2 (sau pool1): {x.shape}")
        
        # Qua lớp conv2 (chỉ conv2, không qua relu hay pool)
        conv2_output = model.conv2(x)
        print(f"Kích thước đầu ra của conv2: {conv2_output.shape}")

    # --- 3. Trực quan hóa các feature map từ lớp conv2 ---
    # Lấy tensor đầu ra của conv2, loại bỏ chiều batch
    feature_maps = conv2_output.squeeze(0)

    # Chúng ta có 32 bộ lọc (out_channels=32), nên sẽ có 32 feature map
    # Ta sẽ vẽ chúng trên một lưới 8x4
    num_filters = feature_maps.shape[0]
    fig, axes = plt.subplots(8, 4, figsize=(12, 24)) # 8 hàng, 4 cột
    fig.suptitle('Đầu ra từ 32 bộ lọc của lớp Conv2', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_filters: # num_filters là 32
            # Lấy ra feature map thứ i
            feature_map = feature_maps[i]
            
            # Vẽ feature map
            ax.imshow(feature_map.numpy(), cmap='gray')
            ax.set_title(f'Bộ lọc (Filter) #{i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Đường dẫn đến ảnh bạn muốn test
    IMAGE_PATH = r"data/processed/train/ky_hieu_tich/nhan_0_1_aug_7.png"

    # --- Khởi tạo model và các phép biến đổi ---
    cnn_model = SimpleCNN(num_classes=3)
    
    # Các phép biến đổi phải GIỐNG HỆT như trong file train_model.py
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    visualize_second_layer_output(model=cnn_model, image_path=IMAGE_PATH, transform=data_transforms) 