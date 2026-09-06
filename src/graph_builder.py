"""
Music Structure Graph Construction Pipeline
============================================
Constructs relational music graphs G = (V, E) from extracted audio segment features
according to the CSE425 project specification:
- Nodes (V): Audio time segments with features h_i^(0).
- Edges (E):
    1. Temporal adjacency edges (i <-> i+1).
    2. Feature similarity edges: Pairwise cosine similarity > tau (similarity_threshold).
- PyTorch Geometric (PyG) Data and Batch structure for GNN message passing.

Hyperparameters and thresholds are loaded dynamically from config.yaml.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml

import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.audio_features import AudioFeatureExtractor, generate_synthetic_audio
except ModuleNotFoundError:
    from audio_features import AudioFeatureExtractor, generate_synthetic_audio

# ---------------------------------------------------------------------------
# Logging & Seed Utilities (AGENTS.md Compliance)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graph_builder")


def set_seed(seed: int = 42) -> None:
    """Set random seed across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to: %d", seed)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load project configuration from config.yaml."""
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Graph Construction Functions
# ---------------------------------------------------------------------------
def build_segment_graph(
    segment_features: np.ndarray,
    similarity_threshold: float = 0.7,
    add_temporal_edges: bool = True,
    add_self_loops: bool = True,
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    track_id: Optional[str] = None,
) -> Data:
    """
    Build a PyTorch Geometric Data object for a single music track's segments.

    Args:
        segment_features: Node features of shape (num_segments, feature_dim).
        similarity_threshold: Cosine similarity cutoff tau to establish similarity edges.
        add_temporal_edges: Whether to connect consecutive segments i and i+1.
        add_self_loops: Whether to include self-loop edges (i, i).
        labels: Optional multi-hot or continuous target vector y.
        track_id: Optional identifier for the music track.

    Returns:
        PyG Data object containing x, edge_index, edge_attr, and optional y.
    """
    num_nodes = segment_features.shape[0]
    x = torch.tensor(segment_features, dtype=torch.float32)

    # Compute pairwise cosine similarity matrix
    norms = np.linalg.norm(segment_features, axis=1, keepdims=True) + 1e-8
    norm_feats = segment_features / norms
    sim_matrix = np.dot(norm_feats, norm_feats.T)  # shape: (N, N)

    edge_sources: List[int] = []
    edge_targets: List[int] = []
    edge_weights: List[float] = []
    edge_set = set()

    # 1. Temporal Adjacency Edges (i <-> i+1)
    if add_temporal_edges and num_nodes > 1:
        for i in range(num_nodes - 1):
            # Forward edge
            edge_sources.append(i)
            edge_targets.append(i + 1)
            edge_weights.append(float(sim_matrix[i, i + 1]))
            edge_set.add((i, i + 1))
            # Backward edge (undirected graph)
            edge_sources.append(i + 1)
            edge_targets.append(i)
            edge_weights.append(float(sim_matrix[i + 1, i]))
            edge_set.add((i + 1, i))

    # 2. Similarity Edges: cos(h_i, h_j) > tau for i != j
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            sim_val = float(sim_matrix[i, j])
            if sim_val >= similarity_threshold:
                if (i, j) not in edge_set:
                    edge_sources.append(i)
                    edge_targets.append(j)
                    edge_weights.append(sim_val)
                    edge_set.add((i, j))
                if (j, i) not in edge_set:
                    edge_sources.append(j)
                    edge_targets.append(i)
                    edge_weights.append(sim_val)
                    edge_set.add((j, i))

    # 3. Self-loops: Ensure message passing functions even with isolated nodes
    if add_self_loops:
        for i in range(num_nodes):
            if (i, i) not in edge_set:
                edge_sources.append(i)
                edge_targets.append(i)
                edge_weights.append(1.0)
                edge_set.add((i, i))

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
    if track_id is not None:
        data.track_id = track_id

    return data


# ---------------------------------------------------------------------------
# Music Graph Dataset Skeleton
# ---------------------------------------------------------------------------
class MusicGraphDataset(Dataset):
    """
    Dataset class for PyTorch Geometric music segment graphs.

    Includes TODO placeholders for loading preprocessed graph files (.pt)
    from data/processed or generating them from raw audio once downloaded.
    Provides synthetic fallback graphs for local offline pipeline testing.
    """

    def __init__(
        self,
        split: str = "train",
        config_path: str = "config.yaml",
        num_labels: int = 50,
        num_samples: int = 8,
        use_dummy_if_missing: bool = True,
    ):
        self.split = split
        self.config = load_config(config_path)
        self.num_labels = num_labels
        self.use_dummy_if_missing = use_dummy_if_missing

        self.sim_threshold = self.config.get("graph", {}).get("similarity_threshold", 0.7)
        self.add_temporal = self.config.get("graph", {}).get("add_temporal_edges", True)

        self.graphs: List[Data] = self._load_or_generate_graphs(num_samples)

    def _load_or_generate_graphs(self, num_samples: int) -> List[Data]:
        """
        Load graphs from data/processed or generate synthetic graphs for testing.
        """
        processed_dir = self.config.get("paths", {}).get("processed", "data/processed")
        split_file = os.path.join(processed_dir, f"{self.split}_graphs.pt")

        # ===================================================================
        # TODO: PREPROCESSED GRAPH DATA LOADING
        # Once audio preprocessing pipeline executes:
        # 1. Check if torch.load(split_file) exists.
        # 2. If present, return loaded list of PyG Data objects.
        # 3. Else, iterate through audio files in data/raw, extract features via
        #    AudioFeatureExtractor, call build_segment_graph(), and save to split_file.
        # ===================================================================
        if os.path.exists(split_file):
            logger.info("Loading preprocessed graphs from: %s", split_file)
            return torch.load(split_file, weights_only=False)

        if self.use_dummy_if_missing:
            logger.warning(
                "Processed graph cache '%s' not found. "
                "Generating %d synthetic graphs for %s split verification.",
                split_file,
                num_samples,
                self.split,
            )
            extractor = AudioFeatureExtractor.from_config("config.yaml")
            synthetic_graphs = []
            seed = self.config.get("seed", 42)

            for i in range(num_samples):
                # Generate varying duration synthetic audio (16s to 40s)
                clip_duration = 16.0 + (i * 3.0)
                audio = generate_synthetic_audio(duration=clip_duration, seed=seed + i)
                node_feats = extractor.extract_segment_features(audio)

                # Synthetic multi-hot tag labels
                np.random.seed(seed + i)
                labels = (np.random.rand(self.num_labels) > 0.85).astype(np.float32)

                graph = build_segment_graph(
                    segment_features=node_feats,
                    similarity_threshold=self.sim_threshold,
                    add_temporal_edges=self.add_temporal,
                    labels=labels,
                    track_id=f"track_synthetic_{i:04d}",
                )
                synthetic_graphs.append(graph)

            return synthetic_graphs
        else:
            raise FileNotFoundError(
                f"Graph data not found at '{split_file}'. Set use_dummy_if_missing=True for pipeline tests."
            )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


def create_graph_dataloader(
    dataset: MusicGraphDataset,
    batch_size: int = 4,
    shuffle: bool = True,
) -> PyGDataLoader:
    """Create a PyTorch Geometric DataLoader that batches multiple graphs into a block-diagonal batch."""
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("Initializing MusicGraphDataset skeleton...")
    dataset = MusicGraphDataset(split="train", config_path="config.yaml", num_samples=4)
    logger.info("Dataset size: %d graphs", len(dataset))

    first_graph = dataset[0]
    logger.info("First graph structure:")
    logger.info("  - Node features x: %s", tuple(first_graph.x.shape))
    logger.info("  - Edge index: %s", tuple(first_graph.edge_index.shape))
    logger.info("  - Edge attributes: %s", tuple(first_graph.edge_attr.shape))
    logger.info("  - Labels y: %s", tuple(first_graph.y.shape))

    # Check PyG DataLoader batching
    loader = create_graph_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    logger.info("PyG Batch verification:")
    logger.info("  - Total batched nodes: %d", batch.num_nodes)
    logger.info("  - Total batched edges: %d", batch.num_edges)
    logger.info("  - Batch assignment vector shape: %s", tuple(batch.batch.shape))
    logger.info("  - Batched targets y shape: %s", tuple(batch.y.shape))

    assert batch.x.ndim == 2, "Node feature tensor must be 2D"
    assert batch.edge_index.shape[0] == 2, "Edge index must have 2 rows (source, target)"
    assert batch.y.shape == (2, 50), f"Expected batched targets shape (2, 50), got {tuple(batch.y.shape)}"
    logger.info("Graph builder sanity check PASSED successfully.")
