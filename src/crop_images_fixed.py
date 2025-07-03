import json
import os
import cv2
import numpy as np

def crop_images_center_format(json_file, image_file, output_folder):
    """
    Cắt ảnh dựa trên thông tin bounding box từ file JSON
    Giả định x,y là tâm của bounding box
    
    Args:
        json_file: Đường dẫn đến file JSON chứa thông tin bounding box
        image_file: Đường dẫn đến file ảnh gốc
        output_folder: Thư mục lưu các ảnh đã cắt
    """
    
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục: {output_folder}")
    
    # Đọc thông tin từ file JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Đọc ảnh gốc
    image = cv2.imread(image_file)
    if image is None:
        print(f"Lỗi: Không thể đọc file ảnh {image_file}")
        return
    
    print(f"Kích thước ảnh gốc: {image.shape[1]}x{image.shape[0]}")
    print(f"Số lượng đối tượng cần cắt: {len(data['boxes'])}")
    
    # Duyệt qua từng bounding box
    for idx, box in enumerate(data['boxes']):
        # Lấy thông tin bounding box
        box_id = box['id']
        label = box['label']
        cx = float(box['x'])  # Tâm x
        cy = float(box['y'])  # Tâm y
        width = float(box['width'])
        height = float(box['height'])
        
        # Tính tọa độ góc từ tâm
        x1 = int(cx - width/2)
        y1 = int(cy - height/2)
        x2 = int(cx + width/2)
        y2 = int(cy + height/2)
        
        # Đảm bảo tọa độ nằm trong phạm vi ảnh
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        # Cắt ảnh
        cropped_image = image[y1:y2, x1:x2]
        
        # Tạo tên file output
        filename = f"{label}_{idx}_{box_id}.jpg"
        output_path = os.path.join(output_folder, filename)
        
        # Lưu ảnh đã cắt
        cv2.imwrite(output_path, cropped_image)
        
        print(f"Đã lưu: {filename} (kích thước: {cropped_image.shape[1]}x{cropped_image.shape[0]})")
    
    print(f"\nHoàn thành! Đã cắt và lưu {len(data['boxes'])} ảnh vào thư mục '{output_folder}'")

def visualize_bounding_boxes_center(json_file, image_file, output_file="data/interim/visualization_center.jpg"):
    """
    Vẽ các bounding box lên ảnh gốc để kiểm tra (với x,y là tâm)
    """
    # Đọc thông tin từ file JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Đọc ảnh gốc
    image = cv2.imread(image_file)
    if image is None:
        print(f"Lỗi: Không thể đọc file ảnh {image_file}")
        return
    
    # Chuyển từ BGR sang RGB (OpenCV dùng BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Vẽ bounding boxes
    for idx, box in enumerate(data['boxes']):
        cx = float(box['x'])  # Tâm x
        cy = float(box['y'])  # Tâm y
        width = float(box['width'])
        height = float(box['height'])
        
        # Tính tọa độ góc từ tâm
        x1 = int(cx - width/2)
        y1 = int(cy - height/2)
        x2 = int(cx + width/2)
        y2 = int(cy + height/2)
        
        # Vẽ hình chữ nhật
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ điểm tâm
        cv2.circle(image_rgb, (int(cx), int(cy)), 3, (255, 0, 0), -1)
        
        # Thêm label
        label = f"{box['label']}_{idx}_{box['id']}"
        cv2.putText(image_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Chuyển lại sang BGR và lưu
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, image_bgr)
    print(f"Đã lưu ảnh với bounding boxes: {output_file}")

def main():
    # Đường dẫn các file
    json_file = "data/interim/annotation_data_tong_sigma.txt"  # File JSON chứa thông tin bounding box
    image_file = "data/raw/ky_hieu_tong_sigma.png"  # File ảnh gốc
    output_folder = "data/processed/ky_hieu_tong_sigma_images"  # Thư mục lưu ảnh đã cắt
    
    # Vẽ visualization để kiểm tra
    visualize_bounding_boxes_center(json_file, image_file)
    
    # Thực hiện cắt ảnh
    crop_images_center_format(json_file, image_file, output_folder)

if __name__ == "__main__":
    main() 