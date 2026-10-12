import torch
from dataclasses import dataclass

@dataclass
class AEConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 14 * 14
    latent_dim: int = 7 * 7
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model_name: str = f"VariationalAutoencoder_MNIST_E{epochs}_B{batch_size}"
