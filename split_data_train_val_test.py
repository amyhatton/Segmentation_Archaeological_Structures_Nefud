import os
import shutil
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

data_dir = "data/bing_300m_1024pix_semantic_segmentation"
files = os.listdir(data_dir)

image_files = [f for f in files if f.startswith("image")]

labels = [f.split("_")[1] for f in image_files]

image_files = np.array(image_files)
labels = np.array(labels)

split_ratios = (0.7,0.2,0.1) #train, val ,test

#split and copy
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=split_ratios[2], random_state=42)
train_val_idx, test_idx = next(sss1.split(image_files, labels))

train_val_files = image_files[train_val_idx]
train_val_labels = labels[train_val_idx]
test_files = image_files[test_idx]


sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.23, random_state=42) #0.23*90 = ~20
train_idx, val_idx = next(sss2.split(train_val_files, train_val_labels))

train_files = train_val_files[train_idx]
val_files = train_val_files[val_idx]


def copy_files(file_list, src_dir,  dst_dir_images, dst_dir_masks):
    os.makedirs(dst_dir_images, exist_ok=True)
    os.makedirs(dst_dir_masks, exist_ok=True)
    for fname in file_list:
        image_src = os.path.join(src_dir, fname)
        mask_name = fname.replace("image_", "mask_")
        mask_src = os.path.join(src_dir, mask_name)
        shutil.copy(image_src, os.path.join(dst_dir_images, fname))
        shutil.copy(mask_src, os.path.join(dst_dir_masks, mask_name))

split_dir = data_dir + "_split"
train_img_dir = os.path.join(split_dir, "train", "images")
train_mask_dir = os.path.join(split_dir, "train", "masks")
val_img_dir = os.path.join(split_dir, "val", "images")
val_mask_dir = os.path.join(split_dir, "val", "masks")
test_img_dir = os.path.join(split_dir, "test", "images")
test_mask_dir = os.path.join(split_dir, "test", "masks")
copy_files(train_files, data_dir,  train_img_dir, train_mask_dir)
copy_files(val_files, data_dir, val_img_dir, val_mask_dir)
copy_files(test_files, data_dir, test_img_dir, test_mask_dir)



