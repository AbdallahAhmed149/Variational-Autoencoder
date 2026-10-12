from dataset import load_mnist_data
from model import Autoencoder
from utils import load_checkpoint, visualize_reconstructions, extract_latent_vectors, visualize_latent_space
from config import AEConfig

def main():
    device = AEConfig.device
    
    _, test_data = load_mnist_data(AEConfig.batch_size)
    
    model = Autoencoder().to(device)
    model_path = f"models/{AEConfig.model_name}.pth"
    
    try:
        model = load_checkpoint(model, model_path, device)
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Did you run main.py first?")
        return
    
    visualize_reconstructions(model, test_data, device, num_images=10)
    
    latent_vectors, labels = extract_latent_vectors(model, test_data, device, num_samples=5000)
    visualize_latent_space(latent_vectors, labels, method="tsne")
    visualize_latent_space(latent_vectors, labels, method="pca")

if __name__ == "__main__":
    main()