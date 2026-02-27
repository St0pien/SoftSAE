"""Copied from "Interpreting CLIP with Hierarchical Sparse Autoencoders" by Vladimir Zaigrajew, Hubert Baniecki, Przemyslaw Biecek:
https://github.com/WolodjaZ/MSAE/blob/main/sae_naming.py"""


import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

import math
import warnings
from tqdm import tqdm

import torch
import random
import numpy as np
from soft_top_k import SoftTopKSAE

"""
Sparse Autoencoder (SAE) Utilities

This module provides utility functions and classes for training and using
Sparse Autoencoders, including dataset handling, learning rate schedulers,
custom activation functions, and various mathematical operations.
"""


class SAEDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset implementation for Sparse Autoencoders.

    This class loads data from memory-mapped numpy arrays to efficiently handle
    large datasets without loading everything into memory at once. It also
    handles preprocessing like mean centering and normalization.

    The class automatically parses dataset dimensions from the filename,
    which is expected to contain the data shape as the last two underscored
    components (e.g., "dataset_name_10000_768.npy" for 10000 vectors of size 768).

    Args:
        data_path (str): Path to the memory-mapped numpy array file
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        mean_center (bool, optional): Whether to center the data by subtracting the mean.
                                     Defaults to False.
        target_norm (float, optional): Target norm for normalization. If None, uses sqrt(vector_size).
                                     If 0.0, no normalization is applied. Defaults to None.
    """

    def __init__(
        self,
        data_path: str,
        dtype: torch.dtype = torch.float32,
        mean_center: bool = False,
        target_norm: float = None,
    ):
        # Parse vector dimensions from filename
        parts = data_path.split("/")[-1].split(".")[0].split("_")
        self.len, self.vector_size = map(int, parts[-2:])

        # Set core attributes
        self.dtype = dtype
        self.data = np.memmap(
            data_path, dtype="float32", mode="r", shape=(self.len, self.vector_size)
        )

        # Special case for representation files (already preprocessed)
        if "repr" in data_path:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.mean_center = False
            self.scaling_factor = 1.0
            return

        # Set preprocessing configuration
        self.mean_center = mean_center
        self.target_norm = (
            np.sqrt(self.vector_size) if target_norm is None else target_norm
        )

        # Compute statistics if needed
        if self.mean_center or self.target_norm != 0.0:
            self._compute_statistics()
        else:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.scaling_factor = 1.0

    def _compute_statistics(self, batch_size: int = 10000):
        """
        Compute dataset statistics (mean and scaling factor) in memory-efficient batches.

        Args:
            batch_size (int, optional): Number of samples to process at once. Defaults to 10000.
        """
        # Compute mean if mean centering is enabled
        if self.mean_center:
            mean_acc = np.zeros(self.vector_size, dtype=np.float32)
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                mean_acc += np.sum(batch, axis=0)
                total += end - start

            self.mean = torch.from_numpy(mean_acc / total).to(self.dtype)
        else:
            self.mean = torch.zeros(self.vector_size, dtype=self.dtype)

        # Compute scaling factor if normalization is enabled
        if self.target_norm != 0.0:
            squared_norm_sum = 0.0
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                # Center the batch if needed
                batch = batch - self.mean.numpy()
                squared_norm_sum += np.sum(np.square(batch))
                total += end - start

            avg_squared_norm = squared_norm_sum / total
            self.scaling_factor = float(self.target_norm / np.sqrt(avg_squared_norm))
        else:
            self.scaling_factor = 1.0

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.len

    def process_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process data for the autoencoder (subtract mean and apply scaling).

        Args:
            data (torch.Tensor): Input data tensor

        Returns:
            torch.Tensor: Processed data tensor
        """
        data.sub_(self.mean)
        data.mul_(self.scaling_factor)

        return data

    def unprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the processing of data (apply inverse scaling and add mean).

        Args:
            data (torch.Tensor): Input data tensor

        Returns:
            torch.Tensor: Unprocessed data tensor
        """
        data.div_(self.scaling_factor)
        data.add_(self.mean)

        return data

    @torch.no_grad()
    def __getitem__(self, idx):
        """
        Get a preprocessed data sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            torch.Tensor: Preprocessed data sample
        """
        torch_data = torch.tensor(self.data[idx])
        output = self.process_data(torch_data.clone())
        return output.to(self.dtype)


"""
Sparse Autoencoder Interpreter

This script computes similarity matrices between input vectors and decoder features
of a Sparse Autoencoder (SAE) model. These matrices can be used to interpret the
learned features and their relationship to the input data.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the similarity computation script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compute similarity matrices for SAE model interpretation"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the trained SAE model file (.pt)",
    )
    parser.add_argument(
        "-v",
        "--vocab",
        type=str,
        required=True,
        help="Path to the target data file (.npy)",
    )
    parser.add_argument(
        "-p",
        "--path-to-save",
        type=str,
        default=".",
        help="Directory path to save the similarity matrices",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for processing data",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=12,
        help="Number of worker processes for data loading",
    )
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument(
        "--patch-diff",
        default=True,
        help="Remove zero sae activation reconstruction vector",
    )
    parser.add_argument("--verbose", action="store_true", help="Print found matches")
    return parser.parse_args()


def cosine_similarity_matrix(
    A: torch.Tensor, B: torch.Tensor, logit_scale: float = 1.0
) -> torch.Tensor:
    """
    Compute the cosine similarity matrix between two sets of vectors.

    This function normalizes both sets of vectors and computes their dot product,
    resulting in a matrix of cosine similarities between all pairs of vectors.

    Args:
        A (torch.Tensor): First set of vectors, shape [n, feature_dim]
        B (torch.Tensor): Second set of vectors, shape [m, feature_dim]
        logit_scale (float): Logit scale for CLIP models it is 100 but for default we set it to 1.0

    Returns:
        torch.Tensor: Cosine similarity matrix of shape [n, m], where each element [i,j]
                     represents the cosine similarity between vector A[i] and B[j]
    """
    # Normalize the vectors to unit length
    A_normalized = A / A.norm(dim=1, keepdim=True)
    B_normalized = B / B.norm(dim=1, keepdim=True)

    # Calculate cosine similarity matrix
    similarity = logit_scale * A_normalized @ B_normalized.t()
    return similarity


def compute_similarities(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    patch_diff: bool = True,
    num_workers: int = 6,
) -> np.ndarray:
    """
    Compute similarity matrices between dataset vectors and decoder features.

    This function:
    1. Processes the dataset in batches
    2. Computes two types of similarity matrices:
       - Standard: between input vectors and decoder features
       - Bias-adjusted: between input vectors and decoder features + bias
    3. Tracks agreement between top matches in both approaches

    Args:
        model (torch.nn.Module): The trained SAE model
        dataset (torch.utils.data.Dataset): Dataset containing input vectors
        batch_size (int): Batch size for processing
        num_workers (int): Number of data loading worker processes
        patch_diff (bool): In the `https://arxiv.org/abs/2407.14499` paper they didn't do it
                            however we find it more interpretable (default: True)

    Returns:
        tuple: (standard_similarity_matrix, bias_similarity_matrix)
            - standard_similarity_matrix: Cosine similarities without bias, shape [n_inputs, n_features]
    """
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get decoder search space (the learned feature vectors)
    decoded_search_space = model.decoder.weight.detach().T + model.b_dec
    decoded_search_space = model.denormalize(decoded_search_space)
    if patch_diff:
        zero_space = model.decode(
            torch.zeros(
                1,
                model.dict_size,
                dtype=decoded_search_space.dtype,
                device=decoded_search_space.device,
            )
        )
        decoded_search_space -= zero_space
    logger.info(
        f"Input features: {len(dataset)}, Decoder features: {decoded_search_space.shape[0]}"
    )

    # Initialize storage for similarity scores
    standard_similarities = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Computing similarities", unit="batch"):
            # In the paper we used `input_processed, _ = model.model.preprocess(batch_data.to(device))`
            # However due to not perfect reconstruction of the model, we simply use the CLIP text
            # Compute similarity
            std_sim = cosine_similarity_matrix(
                batch_data.to(decoded_search_space.device), decoded_search_space
            )
            standard_similarities.append(std_sim.cpu().numpy())

    # Convert lists of batch results to complete matrices
    standard_matrix = np.concatenate(standard_similarities, axis=0).astype(np.float32)

    # Log results
    logger.info("Similarity computation complete")

    # Validate output shapes
    expected_shape = (len(dataset), decoded_search_space.shape[0])
    assert (
        standard_matrix.shape == expected_shape
    ), f"Standard similarity shape mismatch: {standard_matrix.shape} vs expected {expected_shape}"

    return standard_matrix


def main(args):
    """
    Main function to compute and save similarity matrices.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set random seed for reproducibility

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device is None
        else torch.device(args.device)
    )

    # Load the trained SAE model
    model = SoftTopKSAE.from_pretrained(args.model, device=device)
    logger.info("Model loaded")

    # Load the vocabulary dataset
    dataset = SAEDataset(args.vocab, mean_center=False, target_norm=0.0)
    logger.info(f"Vocabulary dataset loaded with {len(dataset)}")

    # Compute both types of similarity matrices
    standard_matrix = compute_similarities(
        model,
        dataset,
        patch_diff=args.patch_diff,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Prepare filenames for saving results
    data_path = os.path.join(
        args.path_to_save,
        f"Concept_Interpreter_{os.path.basename(args.model)}_{os.path.basename(args.vocab)}.npy".replace(
            "/", "~"
        )
        .replace(".pth", "")
        .replace(".pt", "")
        .replace(".npy", ""),
    )
    # Save the matrix
    np.save(data_path, standard_matrix)
    logger.info(f"Successfully saved similarity matrix to {data_path}")

    # Print per vocab best match
    if args.verbose:
        for vocab_id in range(standard_matrix.shape[0]):
            # Get the index of the most similar feature for each input vector
            best_match_index = np.argmax(standard_matrix[vocab_id])
            # Get the similarity score
            best_match_score = standard_matrix[vocab_id, best_match_index]
            # Print the result
            logger.info(
                f"Vocab ID: {vocab_id}/{standard_matrix.shape[0]}({float(vocab_id)/standard_matrix.shape[0]:.2f}), Best match SAE index: {best_match_index}, Score: {best_match_score}"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
