import os
import shutil

# Define source directory (current directory) and destination directory
source_dir = os.getcwd()
dest_dir = os.path.join(source_dir, 'exp2_v4_pdf_result')
# remove dest_dir if exists
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Walk through all subdirectories
for root, dirs, files in os.walk(source_dir):
    # Skip the destination directory itself
    if root == dest_dir:
        continue

    # Process each file
    for file in files:
        if file.lower().endswith('v4_0727.pdf'):
            # Get source and destination paths
            source_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, file)

            # If a file with same name exists, add a suffix
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(file)
                dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                counter += 1

            # Copy the file
            shutil.copy2(source_path, dest_path)
