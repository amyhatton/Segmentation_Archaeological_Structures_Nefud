import torch
import torch.utils
import torch.utils.data
import logging
import lightning as L
#import mlflow 
#import mlflow.pytorch 
#from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.loggers import TensorBoardLogger
#from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, Timer
from torchinfo import summary
import psutil
import time
import os
import segmentation_models_pytorch as smp
from model import ExampleSegment
from data_module import SegmentationDataModule
from dataset import load_dataset
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
#from finetuning_scheduler import FinetuningScheduler


def main(learning_rate, num_channels, batch_size, num_classes, max_epochs, arch, encoder, encoder_weights, decoder_dropout, data_dir, prefix):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
        devices = 1
        strategy = "auto"

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        strategy = "ddp_find_unused_parameters_true" if devices > 1 else None

    else:
        device = torch.device("cpu")
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
    
    print("Device count:", torch.cuda.device_count())
    print("Is cuda available?", torch.cuda.is_available())

    model = ExampleSegment(arch=arch, encoder_name=encoder, encoder_weights=encoder_weights , in_channels=num_channels,
                            out_classes=num_classes, decoder_dropout=decoder_dropout, learning_rate=learning_rate, 
                            batch_size=batch_size, max_epochs=max_epochs, device=device) 
    
    model_summary = summary(model,
                            input_size=(batch_size, num_channels, 256, 256),
                            device=device,
                            col_names=("output_size",
                                    "num_params",
                                    "kernel_size",
                                    "mult_adds",
                                    "trainable",),
                            verbose=2) 

    with open(os.path.join(prefix, "outputs/model_summary.txt"), "w") as text_file:
        print(f"Model Summary: {model_summary}", file=text_file)
    
   
    train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = load_dataset(data_dir=data_dir)

    data_module = SegmentationDataModule(train_imgs, train_masks, val_imgs,
                                        val_masks, test_imgs, test_masks, batch_size)
  
    
    tb_logger = TensorBoardLogger(os.path.join(prefix,'tb_logs'),
                                  name=f"{arch}_model")
    print(tb_logger.log_dir) 
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"
    checkpoint_callback = ModelCheckpoint(
        monitor='dataset_iou/validating',
        mode="max",
        save_top_k=1,   #save best model 
        dirpath=os.path.join(prefix, 'outputs/checkpoints'),
        verbose=True,
        filename=f"{timestr}_{{lr:.3f}}-{{epoch:02d}}-{{validating_dataset_iou:.3f}}_{arch}")
    
    class OverrideEpochStepCallback(Callback):
        def __init__(self) -> None:
            super().__init__()

        def on_train_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule):
            self._log_step_as_current_epoch(trainer, pl_module)

        def on_test_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule):
            self._log_step_as_current_epoch(trainer, pl_module)

        def on_validation_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule):
            self._log_step_as_current_epoch(trainer, pl_module)

        def _log_step_as_current_epoch(self, trainer: Trainer, pl_module: L.LightningModule):
            pl_module.log("step", trainer.current_epoch)

    timer_callback = Timer(duration=None, interval="epoch", verbose=True)

    trainer = Trainer(max_epochs=max_epochs, accelerator=accelerator, 
                        num_nodes=1, strategy=strategy, logger=tb_logger,
                        enable_progress_bar=True,
                        sync_batchnorm=True,
                        callbacks=[checkpoint_callback, OverrideEpochStepCallback(), timer_callback]) #for running on HPC

    trainer.fit(model, data_module)
    train_time = timer_callback.time_elapsed("train")
    val_time = timer_callback.time_elapsed("validate")
    print(f"Total training time: {train_time:.2f} seconds")
    print(f"Total validating time: {val_time:.2f} seconds")

        
if __name__ == "__main__":
    print("I got called from main. Press enter to continue")
    input()
    #setup default values
    LEARNING_RATE=0.0001
    NUM_CHANNELS=3
    BATCH_SIZE= 8
    NUM_CLASSES= 7
    MAX_EPOCHS= 10
    ARCH="Unet" #MAnet
    ENCODER="efficientnet-b3" #resnet18, dpn68
    ENCODER_WEIGHTS="imagenet"
    DROPOUT = 0.2
    DATA_DIR = "data/bing_300m_512pix_semantic_segmentation"
    #DATA_DIR = "data/bing_300m_1024pix_semantic_segmentation"

    main(LEARNING_RATE, NUM_CHANNELS, BATCH_SIZE, NUM_CLASSES, MAX_EPOCHS, ARCH, ENCODER, ENCODER_WEIGHTS, DATA_DIR)
