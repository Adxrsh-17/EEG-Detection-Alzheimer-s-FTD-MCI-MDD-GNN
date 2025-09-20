import mne
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import ewtpy
from scipy.stats import entropy
import mne_connectivity
import json

# --- Configuration ---
PREPROCESSED_METADATA_PATH = Path("../data/preprocessed_metadata.csv")
OUTPUT_GRAPH_DIR = Path("../data/processed/graphs")
OUTPUT_METADATA_PATH = Path("../data/graph_metadata.csv")
OUTPUT_LABEL_MAP_PATH = Path("../data/label_mapping.json")

# --- Feature Engineering Parameters ---
SAMPLING_RATE = 200  # Hz (must match the harmonized data)
BANDS = {
    "Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), 
    "Beta": (13, 30), "Gamma": (30, 45)
}

# --- Helper Functions ---
def shannon_entropy(data: np.ndarray) -> float:
    """Calculates the Shannon Entropy of a signal."""
    counts, _ = np.histogram(data, bins='auto', density=True)
    # Filter out zero probabilities to avoid log(0)
    counts = counts[counts > 0]
    return entropy(counts) # type: ignore

def calculate_node_features(epoch_data: np.ndarray) -> np.ndarray:
    """
    Calculates node features for a single epoch using EWT.
    Input: epoch_data (n_channels, n_times)
    Output: node_features (n_channels, n_features=10)
    """
    n_channels, _ = epoch_data.shape
    node_features = np.zeros((n_channels, len(BANDS) * 2))

    for i in range(n_channels):
        channel_signal = epoch_data[i, :]
        # 1. Apply Empirical Wavelet Transform
        ewt, _, _ = ewtpy.EWT1D(channel_signal, N=len(BANDS))
        
        for band_idx in range(ewt.shape[1]):
            sub_band = ewt[:, band_idx]
            # 2. Calculate features for each band
            # Log-transformed Power (variance)
            log_power = np.log1p(np.var(sub_band))
            # Shannon Entropy
            shan_entropy = shannon_entropy(sub_band)
            
            node_features[i, band_idx * 2] = log_power
            node_features[i, (band_idx * 2) + 1] = shan_entropy
            
    return node_features

def calculate_wpli_connectivity(epochs: mne.Epochs) -> np.ndarray:
    """
    Calculates the broadband wPLI connectivity matrix for all epochs in a file.
    Output: wpli_matrix (n_epochs, n_channels, n_channels)
    """
    conn = mne_connectivity.spectral_connectivity_epochs(
        epochs, method='wpli', mode='multitaper', 
        fmin=BANDS["Delta"][0], fmax=BANDS["Gamma"][1], 
        faverage=True, verbose=False
    )
    # Squeeze to remove the frequency dimension, as we used faverage=True
    return conn.get_data(output='dense').squeeze()

# --- Main Orchestrator ---
def main():
    """Main function to run the graph creation pipeline."""
    print("--- Starting Phase 2: Feature Engineering & Graph Construction ---")
    if not PREPROCESSED_METADATA_PATH.exists():
        print(f"[FATAL] Preprocessed metadata not found. Exiting.")
        sys.exit(1)

    metadata_df = pd.read_csv(PREPROCESSED_METADATA_PATH)
    OUTPUT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Create and save a mapping from string labels to integer indices
    unique_labels = sorted(metadata_df['diagnosis'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    with open(OUTPUT_LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"✅ Saved label mapping to {OUTPUT_LABEL_MAP_PATH}")

    graph_records = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Creating graphs per subject"):
        try:
            epochs = mne.read_epochs(row['preprocessed_file_path'], preload=True, verbose=False)
            
            # Efficiently calculate wPLI for all epochs in the file at once
            wpli_matrices = calculate_wpli_connectivity(epochs)
            # Handle the case where there is only one epoch
            if len(epochs) == 1:
                wpli_matrices = np.expand_dims(wpli_matrices, axis=0)

            # Process each epoch individually for node features and saving
            for i in range(len(epochs)):
                epoch_data = epochs[i].get_data(copy=False).squeeze()
                
                # 1. Calculate node features (X)
                node_features = calculate_node_features(epoch_data)
                
                # 2. Get the pre-calculated connectivity matrix (A)
                adj_matrix = wpli_matrices[i, :, :]
                
                # 3. Convert to PyG format
                edge_index, edge_attr = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))
                x = torch.tensor(node_features, dtype=torch.float)
                y = torch.tensor([label_map[row['diagnosis']]], dtype=torch.long)
                
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                
                # 4. Save the graph object
                graph_filename = f"{Path(row['preprocessed_epo_path']).stem}_graph_{i}.pt"
                graph_filepath = OUTPUT_GRAPH_DIR / graph_filename
                torch.save(graph_data, graph_filepath)
                
                # 5. Create a record for the new graph metadata
                graph_records.append({
                    'subject_id': row['subject_id'],
                    'original_subject_id': row['original_subject_id'],
                    'diagnosis': row['diagnosis'],
                    'graph_file_path': str(graph_filepath.resolve())
                })

        except Exception as e:
            print(f"\n[ERROR] Failed to process subject {row['subject_id']}. Reason: {e}")

    # Save the final metadata file for the graph dataset
    if graph_records:
        graph_df = pd.DataFrame(graph_records)
        graph_df.to_csv(OUTPUT_METADATA_PATH, index=False)
        print("\n--- Graph Construction Complete ---")
        print(f"Successfully created and saved {len(graph_df)} graph files to: {OUTPUT_GRAPH_DIR}")
        print(f"✅ New metadata for the graph dataset saved to: {OUTPUT_METADATA_PATH}")

if __name__ == '__main__':
    main()