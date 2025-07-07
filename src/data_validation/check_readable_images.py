import os
from PIL import Image
from tqdm import tqdm

def check_unreadable_images(root_dir):
    """
    Duyệt qua tất cả các tệp trong một thư mục và các thư mục con của nó,
    kiểm tra xem có tệp ảnh nào không thể đọc được không.

    Args:
        root_dir (str): Đường dẫn đến thư mục gốc cần kiểm tra.
    """
    unreadable_files = []
    
    # Lấy danh sách tất cả các tệp ảnh cần kiểm tra để tqdm có thể hiển thị thanh tiến trình
    image_files_to_check = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files_to_check.append(os.path.join(subdir, file))

    print(f"Bắt đầu kiểm tra {len(image_files_to_check)} tệp ảnh trong thư mục '{root_dir}'...")

    # Sử dụng tqdm để hiển thị thanh tiến trình
    for fpath in tqdm(image_files_to_check, desc="Đang kiểm tra ảnh"):
        try:
            with Image.open(fpath) as img:
                img.verify()  # Kiểm tra tính toàn vẹn của tệp
        except (IOError, SyntaxError) as e:
            unreadable_files.append(fpath)
            # print(f"Lỗi: Không thể đọc tệp ảnh: {fpath} - {e}")

    if not unreadable_files:
        print("\nTuyệt vời! Tất cả các tệp ảnh đều có thể đọc được.")
    else:
        print(f"\nPhát hiện {len(unreadable_files)} tệp ảnh không thể đọc được:")
        for fpath in unreadable_files:
            print(f" - {fpath}")

if __name__ == '__main__':
    # Đường dẫn tới thư mục data/interim của bạn
    # Bạn có thể thay đổi đường dẫn này nếu cần
    data_directory = os.path.join('data', 'interim')
    check_unreadable_images(data_directory) 