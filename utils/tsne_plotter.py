import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import os

def plot_tsne(features: torch.Tensor, labels: torch.Tensor, save_path: str, title: str = "t-SNE Visualization"):
    """
    Performs t-SNE dimensionality reduction and plots the results.

    Args:
        features (torch.Tensor): The features to visualize.
        labels (torch.Tensor): The corresponding labels for the features.
        save_path (str): The path to save the t-SNE plot.
        title (str): The title of the plot.
    """
    if features.shape[0] == 0:
        print("No features to plot for t-SNE.")
        return

    # Ensure features are on CPU and convert to numpy
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    try:
        tsne_results = tsne.fit_transform(features_np)
    except ValueError as e:
        print(f"t-SNE failed: {e}. This might happen if the number of samples is too small or features are not diverse enough.")
        return

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_np, cmap='viridis', s=10)
    plt.colorbar(scatter, ticks=np.unique(labels_np))
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

