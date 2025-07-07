import os
import cv2
import shutil
import albumentations as A
from tqdm import tqdm

# Define the augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=15, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ElasticTransform(p=0.8, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=cv2.BORDER_CONSTANT, value=0),
])

def augment_and_save(image_path, output_dir, num_augmented_images=10):
    """
    Reads an image, applies augmentations N times, and saves the results.
    The background is black (0) and the symbol is white (255), so the padding
    and border mode should use 0 to fill with black.
    """
    try:
        # Albumentations reads images with OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return

        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(num_augmented_images):
            augmented = transform(image=image)
            augmented_image = augmented['image']
            
            output_filename = f"{base_filename}_aug_{i+1}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, augmented_image)

    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")

def process_augmentation(input_base_dir, output_base_dir, num_per_image=10):
    """
    Applies augmentation to all images in subdirectories of the input directory.
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created directory: {output_base_dir}")

    # The input dir has subdirectories for each class
    for class_name in os.listdir(input_base_dir):
        class_input_dir = os.path.join(input_base_dir, class_name)
        class_output_dir = os.path.join(output_base_dir, class_name)

        if not os.path.isdir(class_input_dir):
            continue

        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
            print(f"Created subdirectory: {class_output_dir}")

        image_files = [f for f in os.listdir(class_input_dir) if f.lower().endswith('.png')]
        
        if not image_files:
            print(f"No images found in {class_input_dir}")
            continue
            
        print(f"\nAugmenting directory: {class_input_dir}")
        for filename in tqdm(image_files, desc=f"Augmenting {class_name}"):
            img_path = os.path.join(class_input_dir, filename)
            # Save original image as well
            shutil.copy(img_path, class_output_dir)
            augment_and_save(img_path, class_output_dir, num_augmented_images=num_per_image)


if __name__ == '__main__':
    input_dir = os.path.join('data', 'processed', 'processed_image')
    output_dir = os.path.join('data', 'processed', 'data_augmentation')
    
    # How many augmented versions to create for each original image
    NUM_AUGMENTED_SAMPLES_PER_IMAGE = 10 

    print("Starting data augmentation...")
    process_augmentation(input_dir, output_dir, num_per_image=NUM_AUGMENTED_SAMPLES_PER_IMAGE)
    print("\nData augmentation completed.")
    print(f"Augmented images are saved in: {output_dir}") 