"""
Task 2: Graph Neural Network & Baseline Models
==============================================
Implements:
1. GraphSAGE / GAT Graph Neural Network encoder on music structure graphs.
   - GraphSAGE update: h_i^(l+1) = sigma(W^(l) * CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)))
   - Graph readout: g = (1 / |V|) * sum_{i in V} h_i^(L)
   - Multi-label classification head: y_hat = sigma(W * g + b) with BCEWithLogitsLoss
2. CNN Mel-Spectrogram Baseline (B2):
   - Convolutional baseline operating directly on log-mel spectrograms without graph or text.

Hyperparameters read dynamically from config.yaml as per AGENTS.md rules.
"""

import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
import yaml

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Logging & Seed Utilities (AGENTS.md Compliance)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gnn_model")


def set_seed(seed: int = 42) -> None:
    """Set random seed across libraries for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to: %d", seed)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from config.yaml."""
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
# GNN Encoder Module
# ---------------------------------------------------------------------------
class MusicGNNEncoder(nn.Module):
    """
    Encodes music structure graphs into graph-level representations g in R^d.
    Supports GraphSAGE (default) and GAT convolutional layers.
    """

    def __init__(
        self,
        in_channels: int = 320,
        hidden_dim: int = 128,
        num_layers: int = 2,
        conv_type: str = "sage",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type.lower()
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Build GNN message-passing layers
        for layer_idx in range(num_layers):
            in_dim = in_channels if layer_idx == 0 else hidden_dim
            out_dim = hidden_dim

            if self.conv_type == "sage":
                conv = SAGEConv(in_dim, out_dim, aggr="mean")
            elif self.conv_type == "gat":
                conv = GATConv(in_dim, out_dim, heads=1, concat=False)
            else:
                raise ValueError(f"Unsupported conv_type '{conv_type}'. Choose 'sage' or 'gat'.")

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward message passing through L layers, followed by global mean pooling.

        Args:
            x: Node feature matrix of shape (num_nodes, in_channels).
            edge_index: Graph edge indices of shape (2, num_edges).
            batch: Batch assignment vector of shape (num_nodes,).
            edge_weight: Optional edge weight tensor of shape (num_edges, 1).

        Returns:
            Tuple of:
                - g: Graph-level representation of shape (batch_size, hidden_dim).
                - h: Final node-level representations of shape (num_nodes, hidden_dim).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

        # Graph readout: global mean pooling over node representations
        # g = (1 / |V|) * sum_{i in V} h_i^(L)
        g = global_mean_pool(h, batch)

        return g, h


# ---------------------------------------------------------------------------
# Task 2 GNN Classifier
# ---------------------------------------------------------------------------
class MusicGNNClassifier(nn.Module):
    """
    Task 2: End-to-end GNN classifier on music segment graphs for multi-label tagging.
    """

    def __init__(
        self,
        in_channels: int = 320,
        hidden_dim: int = 128,
        num_layers: int = 2,
        conv_type: str = "sage",
        num_labels: int = 50,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = MusicGNNEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # Linear classification readout: y_hat = sigma(W * g + b)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    @classmethod
    def from_config(
        cls,
        config_path: str = "config.yaml",
        in_channels: int = 320,
        num_labels: int = 50,
    ) -> "MusicGNNClassifier":
        """Instantiate MusicGNNClassifier from config.yaml parameters."""
        config = load_config(config_path)
        gnn_cfg = config.get("gnn", {})
        return cls(
            in_channels=in_channels,
            hidden_dim=gnn_cfg.get("hidden_dim", 128),
            num_layers=gnn_cfg.get("num_layers", 2),
            conv_type=gnn_cfg.get("conv_type", "sage"),
            num_labels=num_labels,
            dropout=gnn_cfg.get("dropout", 0.3),
        )

    def forward(
        self,
        data: Union[Data, Batch],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a PyG Data or batched Data object.
        """
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        g, h = self.encoder(x, edge_index, batch=batch)
        logits = self.classifier(g)
        probabilities = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "graph_embedding": g,
            "node_embeddings": h,
        }

        # Resolve ground truth labels from argument or from data.y
        target_labels = labels if labels is not None else getattr(data, "y", None)
        if target_labels is not None:
            # Ensure target shape matches logits (batch_size, num_labels)
            if target_labels.dim() == 1:
                target_labels = target_labels.unsqueeze(0)
            elif target_labels.shape != logits.shape and target_labels.numel() == logits.numel():
                target_labels = target_labels.view_as(logits)

            loss = self.loss_fn(logits, target_labels.float())
            result["loss"] = loss

        return result


# ---------------------------------------------------------------------------
# Baseline B2: Mel-Spectrogram 2D CNN
# ---------------------------------------------------------------------------
class MelSpectrogramCNNBaseline(nn.Module):
    """
    Baseline B2: 2D Convolutional Neural Network operating directly on
    normalized log-mel spectrograms (without graph or text contextual structure).
    Fulfills the course requirement to compare against baseline models.
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_labels: int = 50,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.num_labels = num_labels

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Spectrogram tensor of shape (batch_size, n_mels, time_frames)
               or (batch_size, 1, n_mels, time_frames).
            labels: Multi-hot target tensor of shape (batch_size, num_labels).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension: (B, 1, M, T)

        feats = self.conv_blocks(x)  # shape: (B, 256, 1, 1)
        pooled = torch.flatten(feats, 1)
        logits = self.classifier(self.dropout(pooled))
        probabilities = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "spectrogram_embedding": pooled,
        }

        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            result["loss"] = loss

        return result


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.graph_builder import MusicGraphDataset, create_graph_dataloader

    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("Initializing MusicGNNClassifier from config.yaml...")
    model = MusicGNNClassifier.from_config("config.yaml", in_channels=320, num_labels=50)
    model.eval()

    logger.info("Loading test graph batch...")
    dataset = MusicGraphDataset(split="train", config_path="config.yaml", num_samples=4)
    loader = create_graph_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        outputs = model(batch)

    logits = outputs["logits"]
    probs = outputs["probabilities"]
    g_emb = outputs["graph_embedding"]
    loss = outputs.get("loss", None)

    logger.info("GNN outputs verified:")
    logger.info("  - Logits shape: %s (expected: [2, 50])", tuple(logits.shape))
    logger.info("  - Graph embedding g shape: %s (expected: [2, %d])", tuple(g_emb.shape), model.hidden_dim)
    if loss is not None:
        logger.info("  - BCEWithLogitsLoss: %.4f", loss.item())

    assert logits.shape == (2, 50), f"Logits shape mismatch: {tuple(logits.shape)}"
    assert g_emb.shape == (2, model.hidden_dim), f"Graph embedding shape mismatch: {tuple(g_emb.shape)}"

    logger.info("Testing Baseline B2 (MelSpectrogramCNNBaseline)...")
    cnn_model = MelSpectrogramCNNBaseline(n_mels=128, num_labels=50)
    cnn_model.eval()

    # Dummy spectrogram batch: (batch_size=2, 1, n_mels=128, time_frames=400)
    dummy_spec = torch.randn(2, 1, 128, 400)
    dummy_labels = torch.randint(0, 2, (2, 50)).float()

    with torch.no_grad():
        cnn_out = cnn_model(dummy_spec, labels=dummy_labels)

    logger.info("CNN Baseline outputs verified:")
    logger.info("  - CNN logits shape: %s", tuple(cnn_out["logits"].shape))
    logger.info("  - CNN loss: %.4f", cnn_out["loss"].item())

    assert cnn_out["logits"].shape == (2, 50)
    assert not torch.isnan(cnn_out["loss"])
    logger.info("Task 2 GNN and CNN Baseline sanity check PASSED successfully.")
