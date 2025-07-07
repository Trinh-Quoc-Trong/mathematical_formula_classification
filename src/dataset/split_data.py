import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

def split_data(base_dir, train_dir, test_dir, train_split=0.8):
    """
    Splits the data into training and testing sets, ensuring class balance.
    """
    # 1. Create output directories if they don't exist
    for d in [train_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d) # Clean up old splits
        os.makedirs(d)
    
    print(f"Created directories: {train_dir}, {test_dir}")

    # 2. Get all class directories and count images per class
    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    images_per_class = defaultdict(list)
    
    for class_name in class_dirs:
        class_path = os.path.join(base_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
        images_per_class[class_name] = [os.path.join(class_path, img) for img in images]

    # 3. Find the minimum number of images in any class to balance the dataset
    if not images_per_class:
        print("No class directories found in the base directory.")
        return

    min_count = min(len(images) for images in images_per_class.values())
    print(f"Balancing all classes to {min_count} images.")

    # 4. Split and copy files for each class
    for class_name, image_paths in tqdm(images_per_class.items(), desc="Splitting classes"):
        # Create class subdirectories in train and test folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Randomly sample min_count images from the current class
        random.shuffle(image_paths)
        sampled_images = image_paths[:min_count]

        # Define split indices
        split_idx = int(len(sampled_images) * train_split)
        
        # Get train and test lists
        train_list = sampled_images[:split_idx]
        test_list = sampled_images[split_idx:]

        # Copy files
        for img_path in train_list:
            shutil.copy(img_path, os.path.join(train_dir, class_name))
            
        for img_path in test_list:
            shutil.copy(img_path, os.path.join(test_dir, class_name))

    print("\nData splitting completed successfully.")

if __name__ == '__main__':
    input_data_dir = os.path.join('data', 'processed', 'data_augmentation')
    
    # Define output directories for the split
    output_base_dir = os.path.join('data', 'processed')
    train_output_dir = os.path.join(output_base_dir, 'train')
    test_output_dir = os.path.join(output_base_dir, 'test')
    
    print("Starting data splitting...")
    split_data(input_data_dir, train_output_dir, test_output_dir)
    print("Data splitting finished.")
    print(f"Training data is in: {train_output_dir}")
    print(f"Testing data is in: {test_output_dir}") 