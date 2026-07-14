import torch
import numpy as np
from sklearn.manifold import TSNE
import os

def save_tsne_data(features: torch.Tensor, labels: torch.Tensor, save_dir: str, file_prefix: str = "tsne_data"):
    """
    Performs t-SNE dimensionality reduction and saves the results (2D coordinates and labels) to .npy files.

    Args:
        features (torch.Tensor): The features to visualize.
        labels (torch.Tensor): The corresponding labels for the features.
        save_dir (str): The directory to save the t-SNE data.
        file_prefix (str): Prefix for the saved file names (e.g., "tsne_data_coordinates.npy", "tsne_data_labels.npy").
    """
    if features.shape[0] == 0:
        print("No features to process for t-SNE data saving.")
        return

    # Ensure features are on CPU and convert to numpy
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Perform t-SNE
    # perplexity should be less than the number of samples. If not, adjust it.
    n_samples = features_np.shape[0]
    perplexity_val = min(30, n_samples - 1) if n_samples > 1 else 1 # Ensure perplexity is valid

    if n_samples <= 1:
        print(f"Not enough samples ({n_samples}) for t-SNE. Skipping data saving.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300)
    try:
        tsne_results = tsne.fit_transform(features_np)
    except ValueError as e:
        print(f"t-SNE failed: {e}. This might happen if the number of samples is too small or features are not diverse enough.")
        return

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define save paths
    coordinates_save_path = os.path.join(save_dir, f"{file_prefix}_coordinates.npy")
    labels_save_path = os.path.join(save_dir, f"{file_prefix}_labels.npy")

    # Save data
    np.save(coordinates_save_path, tsne_results)
    np.save(labels_save_path, labels_np)

    print(f"t-SNE coordinates saved to {coordinates_save_path}")
    print(f"t-SNE labels saved to {labels_save_path}")
