import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torchsummary import summary

from model import SimpleCNN

def plot_metrics(history, save_path):
    """Vẽ và lưu biểu đồ loss và accuracy của quá trình huấn luyện và kiểm định."""
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Vẽ biểu đồ loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Vẽ biểu đồ accuracy
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """Vẽ và lưu ma trận nhầm lẫn (confusion matrix)."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def main():
    # --- Cấu hình (Configuration) ---
    TRAIN_VAL_DIR = 'data/processed/train'      # Thư mục chứa dữ liệu để chia thành train/validation
    REPORTS_DIR = 'reports'                     # Thư mục chính để lưu báo cáo
    FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')  # Thư mục con cho các biểu đồ
    CHECKPOINTS_DIR = 'models/checkpoints'      # Thư mục lưu checkpoint của model sau mỗi epoch
    MODEL_SAVE_PATH = 'models/best_model.pth'   # Đường dẫn lưu model tốt nhất
    
    NUM_CLASSES = 3      # Số lượng lớp cần phân loại
    BATCH_SIZE = 64      # Số lượng ảnh trong một lô (batch)
    NUM_EPOCHS = 20      # Tổng số lần lặp qua toàn bộ tập dữ liệu
    LEARNING_RATE = 0.001 # Tốc độ học của optimizer
    VAL_SPLIT = 0.2      # Tỷ lệ dữ liệu dành cho tập validation (20%)

    # --- Tải và biến đổi dữ liệu (Data Loading & Transformation) ---
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Chuyển ảnh sang ảnh xám (1 kênh)
        transforms.ToTensor(),                       # Chuyển ảnh PIL thành PyTorch Tensor (giá trị 0-1)
        transforms.Normalize([0.5], [0.5])           # Chuẩn hóa tensor về khoảng [-1, 1]
    ])

    print("Loading dataset for training and validation...")
    full_train_dataset = ImageFolder(root=TRAIN_VAL_DIR, transform=data_transforms)
    
    # --- Chia dữ liệu thành tập train và validation ---
    dataset_size = len(full_train_dataset)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    
    print(f"Splitting dataset: {train_size} for training, {val_size} for validation.")
    # `random_split` chia ngẫu nhiên dataset thành hai tập con.
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # `DataLoader` giúp tạo các batch dữ liệu một cách hiệu quả.
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    class_names = full_train_dataset.classes
    print(f"Classes found: {class_names}")
    
    # --- Khởi tạo Model, Hàm mất mát và Trình tối ưu hóa ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    summary(model, (1, 512, 512)) # In tóm tắt kiến trúc model

    # Hàm mất mát `CrossEntropyLoss` phù hợp cho bài toán phân loại đa lớp.
    criterion = nn.CrossEntropyLoss()
    # `Adam` là một optimizer hiệu quả, thường cho kết quả tốt.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Vòng lặp Huấn luyện (Training Loop) ---
    best_val_accuracy = 0.0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Đặt model ở chế độ training (ví dụ: bật Dropout)
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # Xóa gradient của vòng lặp trước
            outputs = model(inputs) # Đưa dữ liệu qua model
            loss = criterion(outputs, labels) # Tính toán loss
            loss.backward() # Lan truyền ngược để tính gradient
            optimizer.step() # Cập nhật trọng số
            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        training_history['train_loss'].append(epoch_train_loss)
        
        # --- Vòng lặp Kiểm định (Validation Loop) ---
        model.eval() # Đặt model ở chế độ evaluation (ví dụ: tắt Dropout)
        all_preds = []
        all_labels = []
        running_val_loss = 0.0
        
        eval_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
        with torch.no_grad(): # Không tính gradient trong quá trình validation
            for inputs, labels in eval_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        
        training_history['val_loss'].append(epoch_val_loss)
        training_history['val_acc'].append(val_accuracy)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # --- Lưu Checkpoint ---
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_train_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # --- Lưu model tốt nhất dựa trên validation accuracy ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with validation accuracy: {best_val_accuracy:.2f}%")
            
            # Tạo và lưu báo cáo cho model tốt nhất trên tập Validation
            report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            df_report.to_csv(os.path.join(REPORTS_DIR, 'validation_classification_report.csv'))
            
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(cm, class_names, os.path.join(FIGURES_DIR, 'validation_confusion_matrix.png'))
            print("Validation reports and confusion matrix for best model saved.")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Vẽ và lưu biểu đồ cuối cùng
    plot_metrics(training_history, os.path.join(FIGURES_DIR, 'training_validation_curves.png'))
    print("Training and validation curves saved.")

if __name__ == '__main__':
    main() 