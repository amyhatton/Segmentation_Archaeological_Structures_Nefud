import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from colorama import Fore
import pandas as pd 
import os
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
#import the segmentation model class from your other file
from model import ExampleSegment
#import the dataset class from your other script
from dataset import SegmentationData
from dataset import load_dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from torchmetrics.classification import MulticlassConfusionMatrix
from glob import glob
from tabulate import tabulate

import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import time

#set up which device to use
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

## INPUTS TO CHANGE ##
num_classes = 7
classes = list(range(0, num_classes ))

class_id_to_name = {

        0:  "Background",
        1: "Cairn" ,
        2 : "Cellular",
        3 : "Mustatil",
        4 : "Pendant",
        5 : "Ringed cairn",
        6 : 'Triangle'
    }

#only using the higher resolution images as they performed so much better
data_dir="data/bing_300m_1024pix_semantic_segmentation_split"
checkpoint_dir = "checkpoints/"

def get_latest_log_dir(model_logdir):
    versions = [d for d in os.listdir(model_logdir) if d.startswith("version_")]
    if not versions:
        return None
    latest = sorted(versions, key=lambda x: int(x.split("_")[1]))[0]
    return os.path.join(model_logdir,latest)

def get_latest_checkpoint(checkpoint_dir, arch_name):
    ckpt_pattern = os.path.join(checkpoint_dir, f"*{arch_name}*.ckpt")
    all_ckpts = glob(ckpt_pattern)
    #sort (they have timestamp at the begining)
    all_ckpts.sort()

    #return the last one (latest)
    return all_ckpts[0]


def load_model_checkpoint(checkpoint_dir):
    model = ExampleSegment.load_from_checkpoint(checkpoint_dir, device=device, strict=False)
    model = model.to(device)
    model.eval
    return model

models = {
    "Segformer": load_model_checkpoint(get_latest_checkpoint(checkpoint_dir, "Segformer")),
    "U-Net": load_model_checkpoint(get_latest_checkpoint(checkpoint_dir, "Unet")),
    "MA-Net": load_model_checkpoint(get_latest_checkpoint(checkpoint_dir, "MAnet"))
}


def compute_confusion_matrix(model, dataloader, num_classes, device=device):
    model.eval()
    model.to(device)
    mcm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    

    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch  # adjust if batch is structured differently
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1).to(device)
            mcm.update(preds, targets)
        
        cm= mcm.compute().cpu().numpy()
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) +1e-8)

    return cm_norm


# Prepare the test dataset and DataLoader
pred_transforms = A.Compose([
            #ToTensorV2(),
            #A.RandomCrop(256, 256, p=0.25),
            #A.ToGray(p=1), #added to test if greyscale is better
            A.Resize(512,512),
            ToTensorV2()
        ])  # Define the same transforms 


#first split off the test set
#train_imgs, test_imgs, train_masks, test_masks = train_test_split(image_paths, mask_paths, test_size = 0.02)
train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = load_dataset(data_dir)
pred_dataset = SegmentationData(test_imgs, test_masks, transform=pred_transforms)
pred_loader = DataLoader(pred_dataset, batch_size=2, shuffle=False, drop_last=True)

# Define a function to visualize the images, masks, and predictions
def plot_image_mask_prediction(image, mask, pred_dict, save_dir):
    """function to plot original image, ground truth mask, and predicted masks of multiple models."""
    # Convert tensors to CPU and detach if they are not already
    image = image.detach().cpu()
    mask = mask.detach().cpu()

    # Permute the image from [C, H, W] to [H, W, C] for plotting
    image = image.permute(1, 2, 0)  # CHW to HWC
   
    present_classes = np.unique(mask)
    print(f"these are the class ids {present_classes}")
    #setup colour palette
    cols = ["#1e77b3",
            "#a6cee3",
            "#b2df8a",
            "#33a02c",
            "#fb9a99",
            "#eb000b",
            "#fdbf6f"]
    classes = [0,1,2,3,4,5,6]
    # Create a dictionary to map class to its color
    class_to_color = {cls: cols[i] for i, cls in enumerate(classes)}
    # Function to apply colors to the mask
    # Function to apply colors to the mask
    def apply_colors_to_mask(mask, class_to_color):
        # Create an empty colored mask (same shape as the mask, but with 3 channels for RGB)
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        
        # Loop through all classes and assign the corresponding color
        for cls, color in class_to_color.items():
            rgb_color = to_rgb(color)  # Convert hex color to RGB tuple (0-1 scale)
            colored_mask[mask == cls] = rgb_color  # Apply class color where the class is present

        return colored_mask
    # Convert ground truth and predicted masks to colored versions
    colored_mask = apply_colors_to_mask(mask, class_to_color)
    
    custom_cmap = ListedColormap( cols)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=custom_cmap(class_id), label=class_id_to_name[class_id])
        for class_id in range(len(present_classes))
]
    
    # Prepare the figure
    n_models = len(pred_dict)
    fig, axs = plt.subplots(1, 2 + n_models, figsize=(5*(2+n_models), 5))

    #Input image
    axs[0].imshow(image)
    axs[0].set_title("Original Image", fontsize = 16)
    axs[0].axis("off")
    
    #Mask
    axs[1].imshow(colored_mask) #cmap="tab20")
    axs[1].set_title("Ground Truth Mask", fontsize = 16)
    axs[1].axis("off")
    
    #Predictions from each model
    for i, (model_name, pred_mask) in enumerate(pred_dict.items(), start=2):
        pred_mask = pred_mask.detach().cpu().squeeze(dim=0).numpy()
        coloured_pred = apply_colors_to_mask(pred_mask, class_to_color)
        axs[i].imshow(coloured_pred)
        axs[i].set_title(f"{model_name} Predicted Mask", fontsize = 16)
        axs[i].axis("off")
    
    #Legend
    # Create custom legend handles using the colors from the `cols` list
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cols[i], markersize=10) for i in range(len(classes))]
    # Labels will now use the class names instead of IDs
    labels = [class_id_to_name[cls] for cls in classes]
    # Move the legend to the right and use class names
    plt.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1.44, 0.8), title="Classes")


    plt.tight_layout
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}_model_predictions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    

pred_dict = {}
# Loop through the test set and make predictions
for batch in pred_loader:
    images, masks = batch  # Unpack the batch (assume test loader returns images and masks)
    
    # Forward pass to get predictions
    with torch.no_grad():
        for name, model in models.items():
            logits = model(images).to(device)
            preds = torch.argmax(logits, dim=1)  # If multi-class, use argmax
            pred_dict[name] = preds[0]
        
        

    # Plot the first image in the batch along with its mask and prediction
    plot_image_mask_prediction(images[0], masks[0], pred_dict, save_dir = "plots/")
    
#fucntion to get the metrics from tensorboard logs

def load_model_metrics(log_dir, metrics_list=None, class_names=None):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    available_tags = ea.Tags()['scalars']

    all_metrics = {}
    detected_classes = list(class_names.keys()) if class_names else []

    if metrics_list is None:
        # default to all metrics found
        metrics_list = set([tag.split("_class_")[0] for tag in available_tags if "_class_" in tag])

    for metric in metrics_list:
        # Per-class metrics
        for c in detected_classes:
            tag = f"{metric}_class_{c}/validating"
            if tag in available_tags:
                events = ea.Scalars(tag)
                events = sorted(events, key=lambda x: x.step)
                values = np.array([e.value for e in events])
                all_metrics[tag] = values  # store each class separately

        # Global metrics
        tag = f"{metric}/validating"
        if tag in available_tags:
            events = ea.Scalars(tag)
            events = sorted(events, key=lambda x: x.step)
            values = np.array([e.value for e in events])
            all_metrics[metric] = values

    return all_metrics, detected_classes

class_cols = {0:"#1e77b3",
            1: "#a6cee3",
            2: "#b2df8a",
            3: "#33a02c",
            4: "#fb9a99",
            5: "#eb000b",
            6: "#fdbf6f"}

def plot_per_class_metric(all_models_metrics, metric_name, save_dir, class_names=None):
    """
    Plot a per-class metric for multiple models.
    
    Args:
        all_models_metrics (dict): {model_name: {metric_name: np.array(epochs, classes)}}
        metric_name (str): e.g., "iou"
        class_names (dict, optional): {class_idx: class_name}
    """
    for model_name, metrics in all_models_metrics.items():

        class_keys = [k for k in metrics.keys() if k.startswith(f"{metric_name}_class_") and k.endswith("/validating")]
    
        if not class_keys:
            print(f"Skipping {model_name}, metric {metric_name} not found.")
            continue
        
        # Sort keys by class index so plotting is consistent
        class_keys = sorted(class_keys, key=lambda x: int(x.split("_class_")[1].split("/")[0]))
        
        num_classes = len(class_keys)

        if class_names is None:
            class_names_list = [f"Class {i}" for i in range(num_classes)]
        elif isinstance(class_names, dict):
            class_names_list = [class_names[i] for i in sorted(class_names.keys())]
        else:
            class_names_list = class_names

        plt.figure(figsize=(10, 6))
        for cls_idx, key in enumerate(class_keys):
            data = metrics[key]  # shape: epochs
            label = class_names_list[cls_idx]
            colour = class_cols[cls_idx] if class_cols else None
            plt.plot(data, label=label, color=colour)

        plt.title(f"{metric_name.upper()} per Class Across Epochs - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.upper())
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"Metric_over_epochs_{model_name}_{metric_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    
def plot_curve(metrics, model_name, save_dir):
    # Get micro-averaged precision and recall values
    precision_vals = metrics[model_name]["precision"]
    recall_vals = metrics[model_name]["recall"]

    plt.figure(figsize=(6, 6))
    plt.plot(recall_vals, precision_vals, marker='o', label=model_name)
    plt.xlabel("Recall (micro)")
    plt.ylabel("Precision (micro)")
    plt.title(f"{model_name}", fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"precision_recall_curve_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, class_names=None,  cmap="Blues", save_path=None):
    """
    Plots a normalized confusion matrix.
    """

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    else:
        # If class_names is a dict, convert to ordered list
        if isinstance(class_names, dict):
            class_names = [class_names[i] for i in sorted(class_names.keys())]

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap=cmap
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name}", fontsize = 16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix figure to {save_path}")

    plt.show()

log_dirs = {
    "SegFormer": "tb_logs/Segformer_model",
    "U-Net": "tb_logs/Unet_model",
    "MA-Net": "tb_logs/MAnet_model"
}


all_models_metrics = {}
summary_results_table = {}

# Initialize and reload
metrics_list = ["iou", "f1", "precision", "recall", "dataset_iou", "dataset_f1_score"]
import re
for model_name, log_path in log_dirs.items():
    
    metrics, detected_classes = load_model_metrics(log_dir= get_latest_log_dir(log_path), metrics_list=metrics_list, class_names=class_id_to_name)
    ckpt_path = get_latest_log_dir(log_path)
    # Extract the epoch number from checkpoint filename
    match = re.search(r"epoch_(\d+)", ckpt_path)
    if match:
        best_epoch = int(match.group(1))
    else:
        best_epoch = -1  # fallback, could also raise an error
    def get_metric_at_epoch(metric_name):
        metric_array = metrics.get(metric_name)
        if metric_array is None:
            return None
        if hasattr(metric_array, "__getitem__"):
            # Metric might be list, np.array, or tensor
            val = metric_array[best_epoch]
            # Convert to float if possible
            if hasattr(val, "item"):
                return float(val.item())
            else:
                return float(val)
        return float(metric_array)  # fallback

    summary_results_table[model_name] = {
        "precision": get_metric_at_epoch("precision"),
        "recall": get_metric_at_epoch("recall"),
        "f1": get_metric_at_epoch("dataset_f1_score"),
        "iou": get_metric_at_epoch("dataset_iou"),
    }

    
    df = pd.DataFrame.from_dict(summary_results_table, orient="index")
    df.index.name = "Model"
    print(tabulate(df.reset_index().values,
                   headers=["Model", "Precision", "Recall", "F1", "IoU"],
                   floatfmt=".3f"))
    log_dir1=get_latest_log_dir(log_path)
    #print(f" this is the log directory {log_dir1}")
    all_models_metrics[model_name] = metrics
    #print(all_models_metrics)
    #print(f"metric keys {metrics.keys()}")

    plot_per_class_metric(all_models_metrics, "iou", class_names=class_id_to_name, save_dir="plots/")
    plot_per_class_metric(all_models_metrics, "f1", class_names=class_id_to_name, save_dir="plots/")
    save_dir_curve = os.path.join("plots/")
    plot_curve(all_models_metrics, model_name, save_dir=save_dir_curve)
    

    #plot_logged_confusion_matrix(log_dir= get_latest_log_dir(log_path), tag="ConfusionMatrixRaw", class_names=class_id_to_name, normalize=True)

for model_name, model in models.items():
    model.to(device)
    model.eval()

    cm = compute_confusion_matrix(model, pred_loader, num_classes=num_classes, device=device)
    save_path = os.path.join("plots/", f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(cm, class_names=class_id_to_name,  save_path=save_path)






     