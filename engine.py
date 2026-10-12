import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb
from config import AEConfig
from model import vae_loss

def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str = AEConfig.device,
    epoch: int = AEConfig.epochs,
    use_wandb: bool = False,
) -> None:
    "Trains the model for one epoch"
    size = len(dataloader.dataset)

    model.train()
    for batch, (images, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward Pass
        reconstructed_imgs, mu, logvar = model(images)
        total_loss = vae_loss(reconstructed_imgs, images, mu, logvar)
        
        # Backward Pass and Backpropagation
        total_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_val = total_loss.item()
            current = (batch + 1) * len(images)
            print(f"Loss: {loss_val:>7f} | [{current:>5d}/{size:>5d}]")

            if use_wandb:
                global_step = epoch * len(dataloader) + batch
                wandb.log({
                    "Train/Loss": loss_val, 
                    "epoch": epoch, 
                    "global_step": global_step
                })


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str = AEConfig.device,
    epoch: int = AEConfig.epochs,
    use_wandb: bool = False,
) -> float:
    """Evaluate the model's performance on the test set."""
    num_batches = len(dataloader)
    total_test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)

            reconstructed_imgs, _, _ = model(images)
            loss = loss_fn(reconstructed_imgs, images)

            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / num_batches
    print(f"Test Set Reconstruction Avg Loss: {avg_test_loss:>8f}")

    if use_wandb:
        wandb.log({
            "Test/Loss": avg_test_loss,
            "epoch": epoch
        })

    return avg_test_loss
