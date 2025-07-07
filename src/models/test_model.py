import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from model import SimpleCNN

def plot_final_confusion_matrix(cm, class_names, save_path):
    """Vẽ và lưu ma trận nhầm lẫn (confusion matrix) cuối cùng."""
    plt.figure(figsize=(10, 8))
    # Sử dụng seaborn để vẽ heatmap cho ma trận nhầm lẫn.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Final Test Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def test_model():
    """
    Tải model đã huấn luyện tốt nhất và đánh giá nó trên tập test cuối cùng.
    Tạo ra một báo cáo hiệu suất chi tiết.
    """
    # --- Cấu hình (Configuration) ---
    TEST_DIR = 'data/processed/test'  # Thư mục chứa dữ liệu test
    MODEL_PATH = 'models/best_model.pth'  # Đường dẫn đến file model tốt nhất đã lưu
    REPORTS_DIR = 'reports/final_test_results'  # Thư mục để lưu báo cáo cuối cùng
    NUM_CLASSES = 3
    BATCH_SIZE = 16

    # --- Tạo thư mục lưu báo cáo ---
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # --- Tải và biến đổi dữ liệu (Data Loading) ---
    # Các phép biến đổi cần phải giống hệt như khi huấn luyện.
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    print("Loading test dataset...")
    test_dataset = ImageFolder(root=TEST_DIR, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")

    # --- Khởi tạo và tải Model (Model Initialization) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    
    # Kiểm tra xem file model có tồn tại không.
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please run the training script first.")
        return
        
    # Tải trọng số (weights) của model tốt nhất đã được lưu.
    model.load_state_dict(torch.load(MODEL_PATH))
    # Chuyển model sang chế độ đánh giá (evaluation mode).
    model.eval()

    # --- Vòng lặp Đánh giá (Evaluation Loop) ---
    all_preds = []
    all_labels = []
    all_probs = []
    misclassified_images = []

    print("Running final evaluation on the test set...")
    # `torch.no_grad()` để tắt việc tính toán gradient, giúp tiết kiệm bộ nhớ và tăng tốc độ.
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            # Lấy chỉ số của lớp có xác suất cao nhất làm dự đoán.
            _, predicted = torch.max(outputs.data, 1)

            # So sánh dự đoán với nhãn thật để tìm ra các ảnh bị nhận diện sai
            for i in range(len(predicted)):
                true_label_idx = labels[i].item()
                pred_label_idx = predicted[i].item()
                
                if pred_label_idx != true_label_idx:
                    # Lấy đường dẫn ảnh gốc từ dataset
                    original_idx = batch_idx * BATCH_SIZE + i
                    if original_idx < len(test_dataset.samples):
                        img_path, _ = test_dataset.samples[original_idx]
                        misclassified_images.append({
                            'file_path': img_path,
                            'true_label': class_names[true_label_idx],
                            'predicted_label': class_names[pred_label_idx]
                        })

            # Lưu lại tất cả các dự đoán và nhãn thật.
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Lưu lại xác suất của các lớp để tính ROC-AUC.
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # --- Tạo và Lưu Báo cáo (Generate and Save Reports) ---
    print("\n--- Final Test Results ---")
    
    # 1. Báo cáo phân loại chi tiết (Accuracy, Precision, Recall, F1-Score)
    # output_dict=True để nhận kết quả dưới dạng dictionary, dễ dàng chuyển thành DataFrame.
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    print("Classification Report:")
    print(df_report)
    df_report.to_csv(os.path.join(REPORTS_DIR, 'final_classification_report.csv'))
    print(f"\nClassification report saved to {os.path.join(REPORTS_DIR, 'final_classification_report.csv')}")

    # 2. Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(all_labels, all_preds)
    plot_final_confusion_matrix(cm, class_names, os.path.join(REPORTS_DIR, 'final_confusion_matrix.png'))

    # 3. Điểm ROC-AUC
    try:
        # Chuyển đổi nhãn sang định dạng one-hot để tính ROC-AUC cho bài toán đa lớp.
        all_labels_binarized = label_binarize(all_labels, classes=range(NUM_CLASSES))
        roc_auc = roc_auc_score(all_labels_binarized, all_probs, multi_class='ovr') # 'ovr' = One-vs-Rest
        print(f"ROC-AUC Score (One-vs-Rest): {roc_auc:.4f}")
        with open(os.path.join(REPORTS_DIR, 'final_roc_auc.txt'), 'w') as f:
            f.write(f"ROC-AUC Score (One-vs-Rest): {roc_auc:.4f}\n")
    except ValueError as e:
        print(f"Could not calculate ROC-AUC score: {e}")

    # 4. Báo cáo các ảnh bị nhận diện sai
    if misclassified_images:
        print(f"\nFound {len(misclassified_images)} misclassified images.")
        df_misclassified = pd.DataFrame(misclassified_images)
        misclassified_path = os.path.join(REPORTS_DIR, 'misclassified_images_report.csv')
        df_misclassified.to_csv(misclassified_path, index=False)
        print(f"Misclassified images report saved to {misclassified_path}")
    else:
        print("\nNo images were misclassified on the test set. Excellent!")

if __name__ == '__main__':
    test_model() 