import torch
import matplotlib.pyplot as plt
from monai.metrics import MAEMetric
from monai.visualize import GradCAM
import argparse
import wandb
from data_loader import *
from model import *

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Arguments for the evaluation script.")
    parser.add_argument('--run_name', type=str, help='Run name for wandb logging.')
    parser.add_argument('--results_dir', type=str, default="./results/", help='Directory to save results.')
    parser.add_argument('--source_test_csv', type=str, help='Path to source test dataset (CSV containing paths to images and labels).')
    parser.add_argument('--model_path', type=str, help='Path to the saved trained model.')
    parser.add_argument('--verbose', type=bool, default=False, help='Flag for verbose output.')
    args = parser.parse_args()

    # If verbose mode is on, print dataset and model information
    if args.verbose:
        print(f"TEST SET: {args.source_test_csv}")
        print(f"MODEL: {args.model_path}")

    # Load the test dataset and its associated data loader
    source_test_ds, source_test_loader = load_test_data(args.source_test_csv)

    # Determine the device (use CUDA GPU if available, otherwise use CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Initialize model and load its weights
    model = build_seq_sfcn().to(device)
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.eval()  # Set model to evaluation mode

    # Initialize Mean Absolute Error metric
    mae_metric = MAEMetric()
    test_mae = 0
    # Lists to store actual ages, predicted ages, and test MAE values
    actual_ages, predicted_ages, test_mae_list = [], [], []

    # Initialize Grad-CAM for two different layers of the model
    grad_cam_12 = GradCAM(nn_module=model, target_layers="features.12")
    grad_cam_20 = GradCAM(nn_module=model, target_layers="features.20")

    # Variables to store cumulative Grad-CAM maps
    total_gradcam_map_12 = 0
    total_gradcam_map_20 = 0

    # Initialize wandb for logging
    run = wandb.init(
        project="YOUR_PROJECT_NAME_HERE",
        name=args.run_name,
        job_type="eval"
    )
    
    # Create a table for logging predictions
    predictions = wandb.Table(columns=["step", "chronological age", "predicted age", "mae"])

    print("Start of testing...")
    for step, batch in enumerate(source_test_loader):
        img, age, sid = batch["img"].to(device), batch["age_label"].to(device), batch["sid"][0]
        img.requires_grad = True

        # Pass the image through the model to get predictions
        output = model(img)

        # Compute Grad-CAM maps for the image
        gc_map_12 = grad_cam_12(x=img)
        gc_map_20 = grad_cam_20(x=img)

        # Accumulate Grad-CAM maps
        total_gradcam_map_12 += gc_map_12
        total_gradcam_map_20 += gc_map_20

        # Compute MAE for the current batch and accumulate
        mae = mae_metric(output, age)
        predictions.add_data(step, age.item(), output.item(), mae.item())
        test_mae += mae.item()
        test_mae_list.append(mae.item())

        print(f"{sid} - Actual Age: {age.item()}, Predicted Age: {output.item()}, MAE: {mae.item()}")

    # Calculate average Grad-CAM maps
    avg_gradcam_map_12 = total_gradcam_map_12 / len(source_test_loader)
    avg_gradcam_map_20 = total_gradcam_map_20 / len(source_test_loader)
    avg_gradcam_map = torch.div(torch.add(avg_gradcam_map_12, avg_gradcam_map_20), 2)

    # Plot and save the average Grad-CAM map
    plt.figure(figsize=(10,8))
    plt.imshow(avg_gradcam_map.cpu().detach().numpy()[0, 0, :, :, 100], cmap='jet_r', alpha=0.7)
    plt.title(f'Average Grad-CAM {args.run_name}')
    plt.colorbar()
    plt.savefig(f"{args.results_dir}/{args.run_name}_smap.png")

    # Compute final test MAE and its standard deviation
    test_mae /= len(source_test_loader)
    test_std = np.std(test_mae_list)

    # Log the results to wandb
    wandb.log({"predictions": predictions})
    test_metrics = wandb.Table(columns=["test_mae", "test_mae_std"])
    test_metrics.add_data(test_mae, test_std)
    wandb.log({"test_metric_results": test_metrics})

    # Plot and save the scatter plot of actual vs predicted ages
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_ages, predicted_ages, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'r')
    plt.savefig(f"{args.results_dir}/{args.run_name}.png")
    print("End of testing")

    wandb.finish()  # End the wandb run