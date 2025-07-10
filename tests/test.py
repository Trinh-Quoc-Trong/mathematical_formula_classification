import os
import shutil
import random
from glob import glob

def create_test_subset(source_dir, dest_dir, sample_percentage=0.4):
    """
    Sao chép một phần trăm ngẫu nhiên các tệp từ thư mục nguồn sang thư mục đích,
    duy trì cấu trúc thư mục con.

    Args:
        source_dir (str): Đường dẫn đến thư mục chứa các thư mục con của lớp (ví dụ: data/processed/train).
        dest_dir (str): Đường dẫn đến thư mục đích để lưu tập con (ví dụ: tests/).
        sample_percentage (float): Tỷ lệ phần trăm tệp cần lấy từ mỗi lớp (ví dụ: 0.4 cho 40%).
    """
    if not os.path.exists(source_dir):
        print(f"Lỗi: Thư mục nguồn không tồn tại: {source_dir}")
        return

    if os.path.exists(dest_dir):
        print(f"Thư mục đích {dest_dir} đã tồn tại. Xóa thư mục cũ...")
        shutil.rmtree(dest_dir)
    
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Đã tạo thư mục đích: {dest_dir}")

    # Lấy danh sách các thư mục con (lớp)
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    if not class_dirs:
        print(f"Không tìm thấy thư mục lớp nào trong {source_dir}")
        return

    print(f"Tìm thấy các lớp: {class_dirs}")

    total_files_copied = 0
    # Lặp qua từng lớp
    for class_name in class_dirs:
        source_class_path = os.path.join(source_dir, class_name)
        dest_class_path = os.path.join(dest_dir, class_name)
        
        os.makedirs(dest_class_path, exist_ok=True)

        # Lấy tất cả các tệp trong thư mục lớp nguồn
        files = glob(os.path.join(source_class_path, '*'))
        
        if not files:
            print(f"Không có tệp nào trong {source_class_path}")
            continue

        # Chọn ngẫu nhiên một số lượng tệp
        num_to_sample = int(len(files) * sample_percentage)
        sampled_files = random.sample(files, num_to_sample)
        
        print(f"Lớp '{class_name}': Lấy {len(sampled_files)} trên tổng số {len(files)} tệp.")

        # Sao chép các tệp đã chọn
        for file_path in sampled_files:
            shutil.copy(file_path, dest_class_path)
        
        total_files_copied += len(sampled_files)

    print(f"\nHoàn tất! Tổng số tệp đã được sao chép: {total_files_copied}")
    print(f"Dữ liệu mẫu đã được lưu vào: {dest_dir}")

if __name__ == '__main__':
    SOURCE_DIRECTORY = 'data/processed/train'
    DESTINATION_DIRECTORY = 'tests/test_subset' # Đổi tên để tránh xung đột với file test.py
    PERCENTAGE = 0.4
    
    create_test_subset(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, PERCENTAGE) 