from run_model import main
import multiprocessing as mp
import sys

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_argument_group(title="Optimization Settings", description="Parameters for the optimization algorithm")
    group.add_argument("--batch-size", default= 32, type=int)
    group.add_argument("--dropout", default= 0.2, type=float)
    group.add_argument("--max-epochs", default= 100, type=int)
    group.add_argument("--lr", default= 0.001, type=float)
    group.add_argument("--model", default= "Unet", type=str)
    group.add_argument("--data-dir", type=str)
    group.add_argument("--prefix", default="" ,type=str, help="Output directory prefix")

    args = parser.parse_args()
    
    if sys.platform == "darwin": #macOS
        mp.set_start_method("spawn", force=True)
    else: #Linux
        mp.set_start_method("fork", force=True)
        
    # Set environment variables for PyTorch model
    NUM_LAYERS=3
    #BATCH_NUMBER=[4,8,16,32,64]
    NUM_CLASSES=7
    #ARCH="Segformer" #"Unet" #MAnet #Segformer
    ENCODER="efficientnet-b3" #resnet18, dpn68
    ENCODER_WEIGHTS="imagenet"
    
    #Call main to run training and validation
    main(args.lr, NUM_LAYERS, args.batch_size, NUM_CLASSES, args.max_epochs, args.model, ENCODER, ENCODER_WEIGHTS, args.dropout, args.data_dir, args.prefix)
