import lightning as L
from typing import Optional
import torch
import torchmetrics
from torchvision import models
import segmentation_models_pytorch as smp
from lightning.pytorch.trainer.states import RunningStage
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassJaccardIndex, MulticlassF1Score, MulticlassPrecisionRecallCurve
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import io
from PIL import Image

class ExampleSegment(L.LightningModule):
    #Initialise the model
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, decoder_dropout, batch_size, max_epochs,
                  device, learning_rate: Optional[int] = None):#, save_debug_images=True, debug_save_freq=1, max_debug_samples = 100):
        #initialising the attributes of the parent class
        super().__init__()
        self.save_hyperparameters()
        self.out_classes =out_classes
        self.class_names = [
                "background",
                "cairn", 
                "cellular",
                "mustatil",
                "pendant",
                "ringed cairn",
                'triangle'
        ]

        self.model = smp.create_model(
            arch = arch,
            encoder_name = encoder_name,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            decoder_dropout = decoder_dropout,
            classes = out_classes,
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1,3,1,1).to(device))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1,3,1,1).to(device))
        
        #initialise the step metrics for debugging
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        

        #choose a loss function
        #weights based on pixels in training data (calculated in class_occurence.py)
        self.loss_fn= smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, smooth = 0.5)
        #self.loss_fn= smp.losses.TverskyLoss('multiclass') 
        #self.loss_fn = torch.nn.CrossEntropyLoss(weight = median_freq_class_weights)
    
        #metrics
        self.f1_score_train = MulticlassF1Score(num_classes=self.number_of_classes, average=None)
        self.f1_score_val =  MulticlassF1Score(num_classes=self.number_of_classes, average=None)
        self.iou_train = MulticlassJaccardIndex(num_classes=self.number_of_classes, average=None)
        self.iou_val = MulticlassJaccardIndex(num_classes=self.number_of_classes, average=None)
        
       
        #Use the normalisation values expected for this model https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html
        #self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        #self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        self.learning_rate = learning_rate
        self.device_type = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        

    def forward(self, x):
        #forward pass Input = batch of images, output = segmentation map
        #Normalise the image (x is the image)
        #had to add .to(device) for the test script to work
        x = (x.to(torch.device(self.device_type)) - self.mean) / self.std
        # returns a dict with only key "out"
        output = self.model(x)
        return output
    
    
    def shared_step(self, batch, stage, batch_idx):
        input_patch, mask_patch = batch
        assert input_patch.ndim == 4 #[batch_size, channels, H, W]
        channels, h, w = input_patch.shape[1:]
        assert h % 32 == 0 and w % 32 == 0
        assert channels == 3

        #convert mask to long (index) tensor for Dice loss
        mask_patch = mask_patch.long()
        assert mask_patch.ndim == 3 #[batch_size, H, W]
        h, w = mask_patch.shape[1:]
        assert h % 32 == 0 and w % 32 == 0
        
        outputs = self.forward(input_patch) #forward pass
        # Ensure the logits mask is contiguous
        outputs = outputs.contiguous()
        #preds = torch.argmax(outputs, dim=1)
        # The Trainer will run .backward(). optimizer.step(), .zero_grad() etc. for you
        #need the mask to be in format batch, width, height - use squeeze to remove dimensions of 1 (eg channels)
        target = torch.squeeze(mask_patch)

        loss = self.loss_fn(outputs, target)
        #removed accuracy becasue it isn't good when classes are imbalanced (eg detecting small object with large background)
        #Convert outputs to probability mask
        prob_mask = outputs.softmax(dim=1)
        #binarise the probability mask
        pred_mask = prob_mask.argmax(dim=1)
        print(f"Predicted mask shape: {pred_mask.shape}")
        print(f"target masks shape: {target.shape}")
       
        # self.log_dict({f"{stage}/loss": loss_to_log}, batch_size=self.batch_size, sync_dist=sync_dist)
        # remap background class 0 -> 255
        pred_mask_no_bg = pred_mask.clone()
        pred_mask_no_bg[pred_mask_no_bg == 0] = -1
        mask_patch_no_bg = mask_patch.clone()
        mask_patch_no_bg[mask_patch_no_bg == 0] = -1

         # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask_no_bg, mask_patch_no_bg, mode="multiclass", ignore_index= -1, num_classes=(self.number_of_classes-1)
        )

        self.val_gt_pixels = []

        if stage == "training":
            self.iou_train.update(pred_mask, target)
            self.f1_score_train.update(pred_mask, target)
        elif stage =="validating":
            self.iou_val.update(pred_mask, target)
            self.f1_score_val.update(pred_mask, target)
            #self.confmat.update(pred_mask, target)
            #self.pr_curve.update(prob_mask, target)
            self.val_gt_pixels.append(target.detach().cpu())

        if stage == "training":
            lr = self.optimizer.param_groups[0]["lr"]
            self.log("learning_rate", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True, reduce_fx='mean')

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "y_true": target.cpu().long(),
            "y_pred": pred_mask.cpu().long()
        }
    
    def shared_epoch_end(self, outputs, stage):
        #Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        losses = torch.stack([x["loss"] for x in outputs])
        avg_loss = losses.mean()

        if stage == 'training':
            iou = self.iou_train.compute()
            f1 = self.f1_score_train.compute()
            self.iou_train.reset()
            self.f1_score_train.reset()
        elif stage == 'validating':
            
            iou = self.iou_val.compute()
            print("VAL IoU compute output:", iou)
            f1 = self.f1_score_val.compute()
            #self.pr_curve.compute()
            #cm = self.confmat.compute()
            self.iou_val.reset()
            self.f1_score_val.reset()
            #self.confmat.reset()
            #self.log_confusion_matrix(cm)
            all_gt_pixels = torch.cat([t.view(-1) for t in self.val_gt_pixels])
            gt_counts = torch.bincount(all_gt_pixels, minlength=self.number_of_classes)
            print(f"Ground truth pixel counts per class:\n{gt_counts}")

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        precision = smp.metrics.precision(tp,fp,fn,tn, reduction = "micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")


        # Determine sync_dist dynamically based on environment
        stage_enum = self.trainer.state.stage
        sync_dist = stage_enum in {RunningStage.SANITY_CHECKING, RunningStage.TESTING, RunningStage.VALIDATING, RunningStage.TRAINING}

        stage = str(stage_enum).split('.')[-1].lower()


        metrics = {
            f"per_image_iou/{stage}": per_image_iou.to(self.device),
            f"dataset_iou/{stage}": dataset_iou.to(self.device),
            f"epoch_loss/{stage}" : avg_loss.to(self.device),
            f"dataset_f1_score/{stage}": dataset_f1.to(self.device),
            f"precision/{stage}":precision.to(self.device),
            f"recall/{stage}":recall.to(self.device)

        }

        for i, iou_class in enumerate(iou):
            metrics[f"iou_class_{i}/{stage}"] = iou_class.to(self.device)
            print(f"class {i} iou value: {iou_class}")

        for i, f1_class in enumerate(f1):
            metrics[f"f1_class_{i}/{stage}"] = f1_class.to(self.device)
            print(f"class {i} f1 value: {f1_class}")
        
        self.log_dict(metrics, prog_bar=True, batch_size=self.batch_size,
                      on_step=False, on_epoch=True, logger=True, sync_dist=True, rank_zero_only=True, reduce_fx='mean') #sync_dist=sync
        
        

    def training_step(self, batch, batch_idx):
        train_loss_info =  self.shared_step(batch, "training", batch_idx)
        #append metrics
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "training")
        #empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "validating", batch_idx)
        self.validation_step_outputs.append(valid_loss_info)

        return valid_loss_info
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "validating")
        self.validation_step_outputs.clear()
        return
    

    def configure_optimizers(self):
        #choose and optimizer - here using stochastic gradient descent
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0 = 10,
                T_mult = 2,
            ),
            "monitor": "dataset_iou/validating",
        }
        return [self.optimizer], [self.scheduler]

    @rank_zero_only
    def log_confusion_matrix(self, cm_tensor, normalize=True):
        cm = cm_tensor.cpu().numpy()

        #normalize
        if normalize:
            cm = cm.astype(float)
            cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)  # row-normalization
        
        # Plot CM
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Best IoU Model)")
        
        # Convert to image
        import io
        from PIL import Image
        import torchvision.transforms as T

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf)
        tensor_image = T.ToTensor()(image)


    
