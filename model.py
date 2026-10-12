import torch
from torch import nn
import torch.nn.functional as F
from config import AEConfig


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dims: int = AEConfig.input_dim,
        hidden_dims: int = AEConfig.hidden_dim,
        latent_dims: int = AEConfig.latent_dim,
    ) -> None:
        super().__init__()

        # Encoder: 784 -> 256 -> 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dims, latent_dims)
        self.fc_logvar = nn.Linear(hidden_dims, latent_dims)

        # Decoder: 64 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, input_dims),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)

        epsilon = torch.randn_like(std)

        z = mu + std * epsilon
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the image from (Batch, 1, 28, 28) to (Batch, 28*28)
        batch_size, channels, height, width = x.size()
        x_flattened = x.view(batch_size, -1)

        latent = self.encoder(x_flattened)

        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)

        x_hat = self.reparameterize(mu, logvar)

        reconstructed_flattend = self.decoder(x_hat)

        # Unflatten back to original image dimensions (Batch, 1, 28, 28)
        reconstructed_image = reconstructed_flattend.view(
            batch_size, channels, height, width
        )

        return reconstructed_image, mu, logvar


# VAE Loss Function
def vae_loss(x_hat, x, mu, logvar):
    """
    Computes: Reconstruction Loss + KL Divergence
    """
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_divergence
