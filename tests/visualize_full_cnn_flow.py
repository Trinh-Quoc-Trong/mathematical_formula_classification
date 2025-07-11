import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import os
import shutil
import math

# Thêm thư mục gốc của dự án vào sys.path để có thể import từ src [[memory:2479219]]
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import kiến trúc model từ file src/models/model.py [[memory:2479219]]
from src.models.model import SimpleCNN

def save_and_show_feature_maps(tensor, stage_name, base_save_dir):
    """
    Hàm phụ trợ để lưu các feature map ra file và hiển thị chúng.
    """
    # Tạo thư mục lưu trữ cho giai đoạn này
    save_dir = os.path.join(base_save_dir, f'stage_{stage_name}')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print(f"Đã tạo thư mục lưu trữ: {save_dir}")

    # Lấy tensor đầu ra, loại bỏ chiều batch
    feature_maps = tensor.squeeze(0)
    num_filters = feature_maps.shape[0]

    # Lưu từng feature map
    for i in range(num_filters):
        feature_map = feature_maps[i].numpy()
        save_path = os.path.join(save_dir, f'filter_{i+1}.png')
        plt.imsave(save_path, feature_map, cmap='gray')
    print(f"Đã lưu {num_filters} feature maps vào thư mục trên.")

    # Hiển thị tất cả feature maps trên một lưới
    grid_size = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f'Đầu ra từ {num_filters} bộ lọc của lớp {stage_name}', fontsize=20)
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            feature_map = feature_maps[i]
            ax.imshow(feature_map.numpy(), cmap='gray')
            ax.set_title(f'Filter #{i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_full_flow(model, image_path, transform, save_path):
    """
    Trực quan hóa luồng dữ liệu qua tất cả các lớp Conv.
    """
    # --- 1. Tải và tiền xử lý ảnh ---
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại: {image_path}")
        return

    image_tensor = transform(image).unsqueeze(0)
    print("--- BẮT ĐẦU ---")
    print(f"Kích thước ảnh đầu vào: {image_tensor.shape}")

    # --- 2. Đưa ảnh qua các lớp ---
    model.eval()
    with torch.no_grad():
        # --- Giai đoạn 1: Conv1 -> ReLU -> Pool1 ---
        print("\n--- GIAI ĐOẠN 1: TÍNH TOÁN QUA KHỐI 1 ---")
        conv1_out = model.conv1(image_tensor)
        print(f"Kích thước sau Conv1: {conv1_out.shape}")
        
        # Áp dụng ReLU và trực quan hóa kết quả sau ReLU [[memory:2479219]]
        relu1_out = F.relu(conv1_out)
        print(f"Kích thước sau ReLU1: {relu1_out.shape}")
        save_and_show_feature_maps(relu1_out, "1_relu_output", save_path)
        
        pool1_out = model.pool1(relu1_out)
        print(f"Kích thước sau Pool1: {pool1_out.shape}")

        # --- Giai đoạn 2: Conv2 -> ReLU -> Pool2 ---
        print("\n--- GIAI ĐOẠN 2: TÍNH TOÁN QUA KHỐI 2 ---")
        conv2_out = model.conv2(pool1_out)
        print(f"Kích thước sau Conv2: {conv2_out.shape}")

        # Áp dụng ReLU và trực quan hóa kết quả sau ReLU [[memory:2479219]]
        relu2_out = F.relu(conv2_out)
        print(f"Kích thước sau ReLU2: {relu2_out.shape}")
        save_and_show_feature_maps(relu2_out, "2_relu_output", save_path)

        pool2_out = model.pool2(relu2_out)
        print(f"Kích thước sau Pool2: {pool2_out.shape}")

        # --- Giai đoạn 3: Conv3 -> ReLU -> Pool3 ---
        print("\n--- GIAI ĐOẠN 3: TÍNH TOÁN QUA KHỐI 3 ---")
        conv3_out = model.conv3(pool2_out)
        print(f"Kích thước sau Conv3: {conv3_out.shape}")
        
        # Áp dụng ReLU và trực quan hóa kết quả sau ReLU [[memory:2479219]]
        relu3_out = F.relu(conv3_out)
        print(f"Kích thước sau ReLU3: {relu3_out.shape}")
        save_and_show_feature_maps(relu3_out, "3_relu_output", save_path)
        
        pool3_out = model.pool3(relu3_out)
        print(f"Kích thước sau Pool3: {pool3_out.shape}")

    print("\n--- KẾT THÚC TRÍCH XUẤT ĐẶC TRƯNG ---")


if __name__ == '__main__':
    IMAGE_PATH = r"data/processed/train/ky_hieu_tich/nhan_0_1_aug_7.png"
    SAVE_RESULTS_DIR = "tests/visualization_results"

    cnn_model = SimpleCNN(num_classes=3)
    
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    visualize_full_flow(
        model=cnn_model, 
        image_path=IMAGE_PATH, 
        transform=data_transforms,
        save_path=SAVE_RESULTS_DIR
    ) 