import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image


def load_dataset(data_dir):
    train_imgs_dir = os.path.join(data_dir, "train", "images")
    train_masks_dir = os.path.join(data_dir, "train", "masks")
    val_imgs_dir =  os.path.join(data_dir, "val", "images")
    val_masks_dir =  os.path.join(data_dir, "val", "masks")
    test_imgs_dir = os.path.join(data_dir, "test", "images")
    test_masks_dir = os.path.join(data_dir, "test", "masks")

    
    filenames_train = sorted(os.listdir(train_imgs_dir))
    filenames_val = sorted(os.listdir(val_imgs_dir))
    filenames_test = sorted(os.listdir(test_imgs_dir))

    print(f"size of training set {len(filenames_train)},size of val set {len(filenames_val)}, size of test set {len(filenames_test)}")
    
    train_imgs = [os.path.join(train_imgs_dir, f) for f in filenames_train] 
    train_masks = [os.path.join(train_masks_dir, f.replace("image", "mask")) for f in filenames_train]
    val_imgs = [os.path.join(val_imgs_dir, f) for f in filenames_val]
    val_masks = [os.path.join(val_masks_dir, f.replace("image", "mask")) for f in filenames_val]
    test_imgs = [os.path.join(test_imgs_dir, f) for f in filenames_test]
    test_masks = [os.path.join(test_masks_dir, f.replace("image", "mask")) for f in filenames_test]
    
    print(f"train image 1 {train_imgs[0]}, train mask 1 {train_masks[0]}")
    print(f"train image 11 {train_imgs[10]}, train mask 11 {train_masks[10]}")
    print(f"train image 51 {train_imgs[50]}, train mask 1 {train_masks[50]}")

    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks

class SegmentationData(Dataset):

    colour_to_class = {

        (255,255,255):0, # "background"
        (55, 126, 184):1, # "cairn" 
        (255, 127, 0):2,  # "cellular"
        #(14, 200, 39):3, # "dwelling"
        #(159, 214, 20):4,# "indeterminate"
        #(75, 220, 220):5, # "keyhole"
        #(202, 60, 60):6, # "kite"
        #(218, 192, 46):7, # "looted cairn"
        (77, 175, 74):3, # "mustatil"
        (152, 78, 163):4, #"pendant"
        #(101, 176, 230):10, # "platform"
        (255, 0, 0):5, #"ringed cairn"
        (255, 255, 51):6 #'triangle'
    }
    #just identifying structures
    # colour_to_class = {

    #     (0,0,0): 0, # "background"
    #     (255,255,255):1  #structure
    # }


    ## Allows you to read in the input and target data. combine them into pairs of tensors 

    ## Args: Dataset (object)

    def __init__(self, image_paths, mask_paths, transform=None):#, test_size: float = 0.1, train_size: float = 0.8, val_size: float = 0.1):

        ## Args: 
        #       image_paths: list of image paths
        #       mask_paths: list of mask paths
        #       transform (bool, optional): applies the defined data transformations

        super(SegmentationData, self).__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    # don't need this step as this is only when downloading data
    #def prepare_data(self):
    #    datasets.VOCSegmentation(root="VOC_data", download=True)
    
    def __len__(self):
        ## Necessary function that returns the length of the dataset
        # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

        #Returns:
        #    _type_: _description_
        number_files_images = len(self.image_paths)
        number_files_masks = len(self.mask_paths)

        if number_files_images == number_files_masks:
            return number_files_images
        else:
            print("number of images is not same as number of masks")

    def __getitem__(self, idx):
        
        ## Necessary function that loads and returns a sample from the dataset at a given index
        # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        # Based on the index it identifies the input and target images location on the disk,
        # reads both images as a numpy array. If transformation argument is True,
        # the defined transformations are applied, else the test transformations are applied

        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        # ~ tilda invert the values in the np array - want to do this because masks are flipped from QGIS
        # .convert("L") - this convert from rgb to single channel image for when doing binary segmentation
        #for multiclass
        mask = np.array(Image.open(self.mask_paths[idx]).convert('RGB'))
        
        # Initialize the output mask (same height and width, single channel)
        mask_remap = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        # Loop through the color-to-class dictionary
        for colour, class_idx in self.colour_to_class.items():
            # Create a mask for pixels matching the current color
            match = np.all(mask == np.array(colour), axis=-1)
            mask_remap[match] = class_idx
        
        mask = mask_remap.astype("float32")

        transformed = self.transform(image=image, mask=mask)
        transformed_i = transformed['image']

        transformed_m = transformed['mask']

        return transformed_i, transformed_m
        

