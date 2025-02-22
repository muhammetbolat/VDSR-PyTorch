import os

# Prepare dataset
os.system("python prepare_dataset.py --images_dir ..\data\T91\original --output_dir ..\data\T91\VDSR\\train --image_size 42 --step 42 --scale 50 --num_workers 10")

# Split train and valid
os.system("python split_train_valid_dataset.py --train_images_dir ..\data\T91\VDSR\\train --valid_images_dir ..\data\T91\VDSR\\valid --valid_samples_ratio 0.1")
