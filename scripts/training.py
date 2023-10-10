import torch
import torch.nn as nn
import wandb

from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MAEMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.data.utils import pad_list_data_collate
from monai.data import decollate_batch

def display_progress(total_steps, verbose=True):
    """Helper function to display progress during training and validation."""
    
    print("=", end="", flush=True)
    
    # Print the step number every 10 steps for clearer progress tracking.
    if total_steps % 10 == 0:
        print(f" {total_steps}", end="", flush=True)

def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir, run, loss, verbose=True):
    """Training function for the neural network."""
    
    # Set the device for computation.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    # Define the loss function based on the user's input.
    if loss.lower() == 'mae':
        loss_object = nn.L1Loss()
    elif loss.lower() == 'mse': 
        loss_object = nn.MSELoss()
    else:
        print("Invalid Loss Function")
        return

    best_val_loss = float('inf')
    
    # Loop through each epoch for training.
    for epoch in range(1, max_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        # Begin training loop for the current epoch.
        print(f"Epoch {epoch}\nTrain:", end="")

        for step, batch in enumerate(train_loader):
            img, age = batch["img"].to(device), batch["age_label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred_glob_age = model(img)
            loss = loss_object(pred_glob_age.float(), age.float())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # Display training progress.
            display_progress(step, verbose)

        train_loss /= len(train_loader)
        
        # Begin validation loop for the current epoch.
        print("\nVal:", end="")
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                img, age = batch["img"].to(device), batch["age_label"].to(device).unsqueeze(1)
                pred_glob_age = model(img)
                loss = loss_object(pred_glob_age.float(), age.float())
                val_loss += loss.item()
                
                # Display validation progress.
                display_progress(step, verbose)
            
            val_loss /= len(val_loader)

        # Print loss values and current learning rate for the epoch.
        print(f"\nTraining epoch {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f} | Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log the losses and learning rate to wandb.
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        # If current validation loss is the best so far, save the model.
        if val_loss < best_val_loss:
            print("Saving model...")
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_model_epoch"] = epoch
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, root_dir + "/" + run.name + "-best-model.pth")
        
        # Adjust the learning rate based on the scheduler.
        scheduler.step()

    return
