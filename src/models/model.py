import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa kiến trúc của mô hình Mạng Nơ-ron Tích chập (CNN).
class SimpleCNN(nn.Module):
    # Hàm khởi tạo, định nghĩa các layer của mạng.
    def __init__(self, num_classes=3):
        # Gọi hàm khởi tạo của lớp cha (nn.Module).
        super(SimpleCNN, self).__init__()
        
        # --- Các lớp Tích chập (Convolutional Layers) ---
        # Mục đích: Trích xuất các đặc trưng (features) từ ảnh như cạnh, góc, hoa văn.
        # Ảnh đầu vào có kích thước: 1 (kênh màu xám) x 512 (cao) x 512 (rộng).
        
        # Lớp Conv2d thứ nhất: 1 kênh đầu vào, 16 kênh đầu ra (16 bộ lọc).
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Lớp MaxPool2d để giảm kích thước chiều cao và rộng đi một nửa.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Lớp Conv2d thứ hai: 16 kênh vào, 32 kênh ra.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Lớp Conv2d thứ ba: 32 kênh vào, 64 kênh ra.
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Các lớp kết nối đầy đủ (Fully Connected Layers) ---
        # Mục đích: Phân loại ảnh dựa trên các đặc trưng đã được trích xuất.
        
        # Kích thước sau khi qua 3 lớp pooling: 64 (kênh) x 64 (cao) x 64 (rộng)
        # self.fc1 chuyển đổi tensor đã được làm phẳng (flattened) thành một vector 512 chiều.
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        # self.fc2 chuyển đổi vector 512 chiều thành đầu ra có số chiều bằng `num_classes`.
        self.fc2 = nn.Linear(512, num_classes)
        # Lớp Dropout để giảm overfitting, với tỷ lệ 50%.
        self.dropout = nn.Dropout(0.5)

    # Hàm `forward` định nghĩa luồng dữ liệu đi qua các lớp đã khai báo.
    def forward(self, x):
        # Luồng qua các lớp Conv và Pooling, với hàm kích hoạt ReLU.
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Làm phẳng (flatten) tensor từ 4D thành 2D để đưa vào lớp Linear.
        x = x.view(-1, 64 * 64 * 64)
        
        # Luồng qua các lớp Linear với ReLU và Dropout.
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Áp dụng dropout trước lớp output cuối cùng.
        x = self.fc2(x)
        return x 