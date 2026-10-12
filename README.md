# Variational Autoencoder

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Weights & Biases](https://img.shields.io/badge/Experiment%20Tracking-W%26B-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai/)

A compact, modular **Variational Autoencoder (VAE)** implemented with PyTorch and trained on the MNIST dataset. The project demonstrates end-to-end representation learning: downloading and preprocessing image data, training a VAE with reconstruction and KL-divergence losses, saving experiments, reconstructing images, and visualizing the learned latent space.

## Overview

This repository provides a practical reference implementation of a fully connected VAE for 28×28 grayscale images. The encoder maps each image into a probabilistic latent representation defined by a mean and log-variance, the reparameterization trick samples a latent vector, and the decoder reconstructs the original image.

The project is designed for experimentation and learning rather than production serving. It separates configuration, data loading, model definition, training/evaluation, persistence, prediction, and visualization into focused Python modules.

> **Note:** The repository does not currently publish benchmark targets such as accuracy, test coverage, throughput, or latency. The README therefore reports the configured experiment dimensions and architecture rather than inventing performance claims.

## Key Features & Metrics

- **Variational representation learning:** Learns a 49-dimensional latent distribution from flattened 784-dimensional MNIST images.
- **Configurable training:** Supports command-line overrides for batch size, epoch count, and learning rate.
- **Reconstruction and latent-space analysis:** Generates reconstruction comparisons and 2D latent-space visualizations using t-SNE and PCA.
- **Experiment persistence:** Saves model weights as `.pth` files and experiment metadata as formatted JSON files.
- **Experiment tracking:** Integrates with Weights & Biases for training/test loss, gradient, and parameter monitoring.
- **Configured baseline:** 10 epochs, batch size 128, learning rate `1e-3`, AdamW optimization, and automatic CPU/accelerator selection.
- **Dataset:** Uses MNIST, consisting of 28×28 grayscale images normalized to floating-point values in the `[0, 1]` range.
- **Architecture dimensions:** `784 → 196 → 49` encoder pathway and `49 → 196 → 784` decoder pathway.

## System Architecture

The project follows a modular, pipeline-oriented design:

```text
MNIST Dataset
     │
     ▼
 dataset.py ──► DataLoaders ──► main.py ──► engine.py
                                      │          │
                                      │          ▼
                                      │       model.py
                                      │          │
                                      ▼          ▼
                              utils.py ◄── Checkpoints + Metrics
                                      │
                                      ▼
                                 predict.py
                                      │
                  Reconstructions + PCA/t-SNE latent visualizations
```

### Data flow

1. `dataset.py` downloads MNIST when necessary and creates training/test `DataLoader` instances.
2. `model.py` defines the VAE encoder, latent mean/log-variance projections, reparameterization step, decoder, and VAE loss.
3. `engine.py` owns the training and evaluation loops, including optimizer updates and optional W&B logging.
4. `main.py` parses runtime arguments, initializes the experiment, trains the model, and saves results.
5. `utils.py` provides checkpoint loading, experiment persistence, reconstruction plotting, and latent-space visualization.
6. `predict.py` loads a saved checkpoint and produces post-training visualizations.

### Design principles

- **Separation of concerns:** Data, model, training, configuration, persistence, and visualization responsibilities are isolated.
- **Configuration over hard-coding:** Baseline hyperparameters are centralized in `AEConfig` and can be overridden from the CLI.
- **Reproducible experimentation:** Model names encode core run parameters, while W&B records training metadata and metrics.
- **Framework-native implementation:** Uses PyTorch modules, automatic differentiation, device placement, and optimized data loading primitives.

## Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python 3.9+ |
| Deep learning | PyTorch, torchvision |
| Dataset | MNIST |
| Optimization | AdamW |
| Experiment tracking | Weights & Biases (`wandb`) |
| Visualization | Matplotlib, Seaborn |
| Dimensionality reduction | scikit-learn PCA and t-SNE |
| Persistence | PyTorch state dictionaries and JSON |
| License | MIT |

## Repository Structure

```text
.
├── config.py       # Default model and training configuration
├── dataset.py      # MNIST download, preprocessing, and DataLoaders
├── engine.py       # Training and evaluation loops
├── main.py         # Training entry point and CLI arguments
├── model.py        # VAE model and loss function
├── predict.py      # Checkpoint loading and visualization entry point
├── utils.py        # Checkpoints, metrics, reconstructions, and latent plots
├── LICENSE         # MIT License
├── metrics/        # Generated metrics and visualization artifacts
└── models/         # Generated PyTorch checkpoints
```

The `models/` and `metrics/` directories are created automatically when an experiment is saved.

## Getting Started

### Prerequisites

- Python 3.9 or newer
- `pip`
- A CPU or a PyTorch-supported accelerator
- Optional: a Weights & Biases account for experiment tracking

### 1. Clone the repository

```bash
git clone https://github.com/AbdallahAhmed149/Variational-Autoencoder.git
cd Variational-Autoencoder
```

### 2. Create and activate a virtual environment

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install a PyTorch build appropriate for your operating system and accelerator by following the official PyTorch installation selector. Then install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install torch torchvision wandb matplotlib seaborn scikit-learn numpy
```

### 4. Configure Weights & Biases

The training script initializes W&B by default. Authenticate before training:

```bash
wandb login
```

If you do not want to use W&B, disable or remove the `wandb.init`, `wandb.watch`, `wandb.log`, `wandb.save`, and `wandb.finish` calls in `main.py`, and pass `use_wandb=False` to the training/evaluation functions as appropriate.

No repository-specific environment variables are currently required.

## Usage

### Train the VAE with the defaults

```bash
python main.py
```

The default configuration is:

- Batch size: `128`
- Epochs: `10`
- Learning rate: `0.001`
- Optimizer: AdamW
- Weight decay: `1e-5`

MNIST is downloaded automatically into `./data` on the first run. After training, the checkpoint is saved under `models/` and experiment metadata is saved under `metrics/`.

### Override training parameters

```bash
python main.py --batch_size 256 --epochs 20 --lr 0.0005
```

### Generate reconstructions and latent-space plots

After a successful training run:

```bash
python predict.py
```

This command:

1. Loads the checkpoint associated with the configured experiment name.
2. Saves an original-versus-reconstructed image grid to `metrics/`.
3. Extracts latent vectors from up to 5,000 test examples.
4. Saves both t-SNE and PCA latent-space visualizations to `metrics/`.

### Programmatic model usage

```python
import torch
from model import Autoencoder
from utils import load_checkpoint

model = Autoencoder()
model = load_checkpoint(model, "models/VariationalAutoencoder_MNIST_E10_B128.pth", "cpu")

with torch.no_grad():
    reconstructed_images, mean, log_variance = model(images)
```

Here, `images` should be a floating-point tensor with shape `(batch_size, 1, 28, 28)` and values in `[0, 1]`.

## Outputs

A training run produces artifacts similar to the following:

```text
models/
└── VariationalAutoencoder_MNIST_E10_B128.pth

metrics/
├── VariationalAutoencoder_MNIST_E10_B128_metrics.json
├── reconstructions_VariationalAutoencoder_MNIST_E10_B128.png
├── latent_space_tsne_VariationalAutoencoder_MNIST_E10_B128.png
└── latent_space_pca_VariationalAutoencoder_MNIST_E10_B128.png
```

Generated datasets, checkpoints, metrics, and plots should generally remain untracked unless they are intentionally included as project artifacts.

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository.
2. Create a focused branch:

   ```bash
   git checkout -b feature/your-change
   ```

3. Implement the change and update documentation when behavior or setup changes.
4. Run the training and/or prediction workflow relevant to your change.
5. Commit with a clear message and push your branch.
6. Open a pull request describing the motivation, implementation, validation steps, and any changes to generated artifacts.

For model or training changes, include the relevant hyperparameters and evaluation observations so results can be reproduced.

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for the full license text.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [torchvision](https://pytorch.org/vision/stable/) for the MNIST dataset utilities.
- [Weights & Biases](https://wandb.ai/) for experiment tracking.
