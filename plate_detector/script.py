import os
import shutil

# Paths to the images and labels directories
images_dir = "./datasets/images/train"
labels_dir = "./datasets/labels/train"

# Paths for the output directories
output_images_dir = "./datasets/images/cull"
output_labels_dir = "./datasets/labels/cull"

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Get a sorted list of image files and corresponding annotation files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

# Ensure the dataset is aligned (same number of images and labels with matching names)
if len(image_files) != len(label_files):
    print("Error: Mismatch in the number of images and labels.")
    exit(1)

# Cull to the first 2000
cull_count = 1000
for i, image_file in enumerate(image_files[:cull_count]):
    label_file = image_file.replace('.jpg', '.txt')
    
    if label_file not in label_files:
        print(f"Error: Annotation file missing for {image_file}")
        continue
    
    # Copy the files to the output directories
    shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_images_dir, image_file))
    shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_labels_dir, label_file))

print(f"Successfully culled the first {cull_count} images and their annotations.")
