import torch
import torch.nn as nn
import argparse
import wandb
import numpy as np

from data_loader import load_data
from training import train
from model import build_seq_sfcn
from torch.optim.lr_scheduler import StepLR

# Initialize argument parser
def initialize_parser():
    """Initialize the argument parser for command-line arguments."""
    parser = argparse.ArgumentParser(description="Training script for FAIMI age prediction.")
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size, number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs')
    parser.add_argument('--run_name', type=str, help='Run name for wandb')
    parser.add_argument('--loss', type=str, default='MAE', help='Loss metric to use')
    parser.add_argument('--results_dir', type=str, default="./results/", help='Directory to store results')
    parser.add_argument('--source_train_csv', type=str, help='Path to source train dataset (images and age labels)')
    parser.add_argument('--source_val_csv', type=str, help='Path to source validation dataset (images and age labels)')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose debugging flag')
    
    return parser

def main():
    """Main function to run the training process."""
    args = initialize_parser().parse_args()

    # Path to store results
    root_dir = args.results_dir
    # Debugging flag
    verbose = args.verbose
    
    # Load data
    source_ds_train, source_train_loader, source_ds_val, source_val_loader = \
        load_data(args.source_train_csv, args.source_val_csv, root_dir, batch=args.batch_size, verbose=verbose)
    
    if verbose: # Log train data information
        print("Train Data:", args.source_train_csv, flush=True)

        # Extract train labels
        source_train_labels = np.array([item['age_label'] for item in source_ds_train])
        print("Source train labels shape:", source_train_labels.shape, flush=True)
        print("Source train labels data type:", source_train_labels.dtype, flush=True)

    # Initialize model, optimizer and scheduler
    model = build_seq_sfcn()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    model = model.cuda()
    
    #Define Scheduler STEP and GAMMA
    scheduler_step = 10  #YOUR SCHEDULER STEP HERE
    scheduler_gamma = 0.5 #YOUR SCHEDULER GAMMA HERE

    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    
    # Initialize wandb
    run = wandb.init(
        project="YOUR_PROJECT_NAME_HERE",
        config={
            "learning_rate_initial": args.learning_rate,
            "batch": args.batch_size,
            "epochs": args.epochs,
            "scheduler_step": scheduler_step,
            "scheduler_gamma": scheduler_gamma, 
            "loss": args.loss
        },
        name=args.run_name,
        job_type="train"
    )

    print("Start of training...", flush=True)
    train(source_train_loader, source_val_loader, model, optimizer, scheduler, args.epochs, root_dir, run, args.loss)
    print("End of training.", flush=True)
    
    wandb.finish()

if __name__ == "__main__":
    main()
