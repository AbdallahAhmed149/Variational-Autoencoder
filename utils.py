import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import AEConfig

def load_checkpoint(model: nn.Module, path: str, device: str) -> nn.Module:
    "Loads a saved model state dictionary."
    model.load_state_dict(torch.load(path, map_location=device))
    
    # CRITICAL: Always set the model to evaluation mode after loading for inference
    model.eval()
    print(f"Model successfully loaded from {path}")
    return model

def save_experiment(model_state: dict, metrics: dict, model_name: str) -> None:
    """Saves the model weights and JSON file with the performance metrices."""
    
    model_dir = "models"
    metrics_dir = "metrics"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    
    torch.save(model_state, model_path)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n--- Experiment Saved ---")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

def visualize_reconstructions(model: nn.Module, test_loader: DataLoader, device: str, num_images: int = 10) -> None:
    """
    Plots original input images against their autoencoder reconstructions.
    """
    model.eval()
    
    # Extract exactly one batch for visualization
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    
    with torch.no_grad():
        reconstructed, _, _ = model(images)

    images_cpu = images.cpu().numpy()
    reconstructed_cpu = reconstructed.cpu().numpy()
    
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(num_images * 2, 4))
    fig.suptitle("Vanilla Autoencoder: Original vs. Reconstructed", fontsize=16)
    
    for i in range(num_images):
        axes[0, i].imshow(images_cpu[i].squeeze(), cmap="gray")
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(reconstructed_cpu[i].squeeze(), cmap="gray")
        axes[1, i].set_title(f"Recon {i + 1}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Saving the Figure
    save_path = f"metrics/reconstructions_{AEConfig.model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def extract_latent_vectors(model: nn.Module, dataloader: DataLoader, device: str, num_samples: int = 5000):
    """
    Passes images through the encoder and collects the latent vectors.
    """
    model.eval()

    latent_vectors = []
    labels_list = []
    samples_collected = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            x_flattened = images.view(batch_size, -1)
            latent = model.encoder(x_flattened)

            latent_vectors.append(latent.cpu().numpy())
            labels_list.append(labels.numpy())

            samples_collected += batch_size
            if samples_collected >= num_samples:
                break

    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels = np.concatenate(labels_list, axis=0)[:num_samples]

    return latent_vectors, labels

def visualize_latent_space(latent_vectors, labels, method="tsne"):
    """
    Reduces the 64D vectors to 2D and plots them.
    """
    print(f"Running {method.upper()} for dimensionality reduction...")

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    latent_2d = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latent_2d[:, 0],
        y=latent_2d[:, 1],
        hue=labels,
        palette=sns.color_palette("tab10", 10),
        legend="full",
        alpha=0.7,
        s=20,
    )

    plt.title(f"Latent Space Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)

    # Saving the Figure
    save_path = f"metrics/latent_space_{method}_{AEConfig.model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()