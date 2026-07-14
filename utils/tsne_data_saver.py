import torch
import numpy as np
import os

def save_tsne_data(features: torch.Tensor, labels: torch.Tensor, save_dir: str, file_prefix: str = ""):
    """
    Saves features and labels to .npy files for later t-SNE visualization or analysis.

    Args:
        features (torch.Tensor): The features to save.
        labels (torch.Tensor): The corresponding labels for the features.
        save_dir (str): The directory to save the .npy files.
    """
    if features.shape[0] == 0:
        print("No features to save for t-SNE.")
        return

    # Ensure features are on CPU and convert to numpy
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define save paths
    prefix_str = f"{file_prefix}_" if file_prefix else ""
    features_save_path = os.path.join(save_dir, f"{prefix_str}features.npy")
    labels_save_path = os.path.join(save_dir, f"{prefix_str}labels.npy")

    # Save the numpy arrays
    np.save(features_save_path, features_np)
    np.save(labels_save_path, labels_np)

    print(f"t-SNE features saved to {features_save_path}")
    print(f"t-SNE labels saved to {labels_save_path}")
