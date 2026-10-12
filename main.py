import os
import time
import argparse
from torch import nn, optim
import wandb

from dataset import load_mnist_data
from model import Autoencoder
from engine import train, test
from utils import save_experiment
from config import AEConfig

def main():
    # Setup Configuration via Argparse
    parser = argparse.ArgumentParser(description="Train Autoencoder Model on MNIST Data")
    parser.add_argument("--batch_size", type=int, default=AEConfig.batch_size, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=AEConfig.epochs, help="Number of epoch to train")
    parser.add_argument("--lr", type=float, default=AEConfig.learning_rate, help="Learning rate")
    args = parser.parse_args()
    
    AEConfig.batch_size = args.batch_size
    AEConfig.epochs = args.epochs
    AEConfig.learning_rate = args.lr
    
    model_name = AEConfig.model_name
    
    # Setup Device
    device = AEConfig.device
    print(f"Using {device} device")
    
    # Initialize Data Pipeline
    print("Loading data...")
    train_loader, test_loader = load_mnist_data(args.batch_size)
    
    # Initialize Model, Loss, and Optimizer
    model = Autoencoder().to(device)
    criterion = nn.BCELoss(reduction="sum")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Initialize Weights & Biases
    run_name = f"Autoencoder_MNIST_E{args.epochs}_B{args.batch_size}_LR{args.lr}"
    wandb.init(
        project="autoencoder-mnist-mastery",
        name=run_name,
        config={
            "architecture": "Variational Autoencoder",
            "dataset": "MNIST",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr
        }
    )
    
    # MAGIC FEATURE: Watch the model to track gradients and parameter distributions!
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Execute Training Loop
    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")

    # Start Tracking Time
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, model, criterion, optimizer, device, epoch, use_wandb=True)
        final_test_loss = test(test_loader, model, criterion, device, epoch, use_wandb=True)
    
    # End Tracking Time
    end_time = time.time()
    training_duration = end_time - start_time
    
    print("Training Complete!")
    
    # Combine our hyperparameters with our actual results
    experiment_data = {
        "architecture": model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "final_test_loss": final_test_loss,
        "training_time_seconds": round(training_duration, 2)
    }
    
    # Saving Model Results
    save_experiment(
        model_state=model.state_dict(),
        model_name=model_name,
        metrics= experiment_data
    )
    
    # Upload the Model to the Cloud
    cloud_model_path = os.path.join("models", f"{model_name}.pth")
    wandb.save(cloud_model_path)
    
    # Close and run
    wandb.finish()

if __name__ == "__main__":
    main()