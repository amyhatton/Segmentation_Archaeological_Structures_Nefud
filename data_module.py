import lightning as L
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional

from dataset import SegmentationData
from torch.utils.data import DataLoader

# Define your lightning module for the model

#### Data Module ####

class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks, batch_size):
        super().__init__()
        self.train_imgs = train_imgs
        self.train_masks = train_masks
        self.val_imgs = val_imgs
        self.val_masks = val_masks
        self.test_imgs = test_imgs
        self.test_masks = test_masks
        self.batch_size = batch_size

        #Define the transformations
        # Example: https://albumentations.ai/docs/examples/pytorch_classification/
        self.train_transform = A.Compose([
            #
            A.RandomCrop(256,256, p=0.25),
            A.HorizontalFlip(p=0.25),
            A.RandomRotate90(p=0.25),
            A.VerticalFlip(p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
            A.CLAHE(clip_limit = 4, p=0.25),
            A.ToGray(p=0.25),
            A.Resize(256, 256),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None):
        
        # stages are fit, validate, test or predict
        self.train_dataset = SegmentationData(self.train_imgs, self.train_masks, transform=self.train_transform)
        self.val_dataset = SegmentationData(self.val_imgs, self.val_masks, transform=self.val_transform)
        self.test_dataset = SegmentationData(self.test_imgs, self.test_masks, transform=self.val_transform)
        

    #hashed for testing images
    #Dataloaders: we want shuffle to be true for training 
    #use drop_last= True to solve issue of mismatched tensor shapes when last batch is smaller (ie sample not divisible by batch size)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=True, num_workers=18, drop_last=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=18,  drop_last=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=18, drop_last=True, persistent_workers=True)
