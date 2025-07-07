import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def preprocess_image(img_path, output_path, size=(512, 512)):
    """
    Applies a series of preprocessing steps to a single image:
    1. Resize with padding to a square.
    2. Convert to grayscale.
    3. Apply Otsu's binarization.
    4. Save the processed image.
    """
    try:
        # Open the image using Pillow
        img = Image.open(img_path).convert("RGB")

        # 1. Resize with padding
        original_width, original_height = img.size
        target_width, target_height = size
        
        ratio = min(target_width/original_width, target_height/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
            
        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new white background image
        new_img = Image.new("L", size, 255)
        
        # Calculate pasting position for centering
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste the resized image
        new_img.paste(img_resized.convert('L'), (paste_x, paste_y))

        # 2. Grayscale is already done by creating a new 'L' mode image
        # and pasting a converted 'L' mode resized image.

        # 3. Binarization using OpenCV and Otsu's method
        img_np = np.array(new_img)
        
        # Use THRESH_BINARY_INV to make symbol white (255) and background black (0)
        _, binarized_img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        final_img = Image.fromarray(binarized_img_np)
        
        # 4. Save the processed image
        final_img.save(output_path)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def process_directories(input_dirs, output_base_dir, size=(512, 512)):
    """
    Processes all images in a list of input directories and saves them to the output directory.
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created directory: {output_base_dir}")

    for input_dir in input_dirs:
        # On Windows, input_dir can be data\\interim\\ky_hieu_tich, basename would be ky_hieu_tich
        # On Linux/Mac, it would be data/interim/ky_hieu_tich
        dir_name = os.path.basename(os.path.normpath(input_dir))
        output_dir = os.path.join(output_base_dir, dir_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created subdirectory: {output_dir}")

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            continue

        print(f"\nProcessing directory: {input_dir}")
        for filename in tqdm(image_files, desc=f"Processing {dir_name}"):
            img_path = os.path.join(input_dir, filename)
            
            # Change extension to .png for consistency
            base_filename = os.path.splitext(filename)[0]
            output_filename = base_filename + ".png"
            output_path = os.path.join(output_dir, output_filename)

            preprocess_image(img_path, output_path, size)

if __name__ == '__main__':
    # Define input directories relative to the workspace root
    base_interim_dir = os.path.join('data', 'interim')
    input_directories = [
        os.path.join(base_interim_dir, 'ky_hieu_tich'),
        os.path.join(base_interim_dir, 'ky_hieu_tong_can'),
        os.path.join(base_interim_dir, 'ky_hieu_tong_sigma_images')
    ]

    # Define the base output directory
    output_base = os.path.join('data', 'processed', 'processed_image')

    # Define the target size for the images
    target_size = (512, 512)

    print("Starting image preprocessing...")
    process_directories(input_directories, output_base, target_size)
    print("\nImage preprocessing completed.")
    print(f"Processed images are saved in: {output_base}") 