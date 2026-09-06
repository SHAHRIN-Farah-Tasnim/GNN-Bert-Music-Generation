"""
Task 4: Cross-Modal MusicCaps Alignment via Contrastive Learning
================================================================
Implements dual-encoder contrastive alignment between audio structure graphs
(GNN) and natural-language text descriptions (DistilBERT) using InfoNCE loss.

Mathematical Formulation:
    g_i = Normalize(W_audio * GNN(G_i))
    t_i = Normalize(W_text * BERT_CLS(caption_i))
    S_ij = (g_i^T * t_j) / tau

    L_NCE = - (1 / 2N) * sum_{i=1}^N [
        log( exp(S_ii) / sum_j exp(S_ij) ) +
        log( exp(S_ii) / sum_j exp(S_ji) )
    ]

Cross-Modal Retrieval Metrics:
    - Caption -> Audio: R@1, R@5, R@10, MRR
    - Audio -> Caption: R@1, R@5, R@10, MRR

Hyperparameters read dynamically from config.yaml as per AGENTS.md rules.
"""

import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from transformers import AutoModel, AutoTokenizer
import yaml

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn_model import MusicGNNEncoder

# ---------------------------------------------------------------------------
# Logging & Seed Utilities (AGENTS.md Compliance)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("contrastive")


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
# Dual-Encoder Contrastive Model
# ---------------------------------------------------------------------------
class MusicContrastiveDualEncoder(nn.Module):
    """
    Task 4: Dual-Encoder GNN-BERT architecture projecting audio graphs and
    natural-language captions into a shared embedding space.
    """

    def __init__(
        self,
        in_channels: int = 320,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        conv_type: str = "sage",
        bert_model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = True,
        proj_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # 1. Audio Graph Encoder (GNN)
        self.audio_encoder = MusicGNNEncoder(
            in_channels=in_channels,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            conv_type=conv_type,
            dropout=dropout,
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(gnn_hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        # 2. Text Caption Encoder (BERT)
        logger.info("Loading BERT encoder for contrastive alignment: %s", bert_model_name)
        self.text_encoder = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_dim = self.text_encoder.config.hidden_size

        if freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            logger.info("BERT text encoder parameters frozen.")

        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        # Learnable log-temperature parameter initialized to log(1 / temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))

    @classmethod
    def from_config(
        cls,
        config_path: str = "config.yaml",
        in_channels: int = 320,
        proj_dim: int = 128,
        temperature: float = 0.07,
    ) -> "MusicContrastiveDualEncoder":
        """Instantiate dual encoder using parameters from config.yaml."""
        config = load_config(config_path)
        gnn_cfg = config.get("gnn", {})
        bert_cfg = config.get("bert", {})

        return cls(
            in_channels=in_channels,
            gnn_hidden_dim=gnn_cfg.get("hidden_dim", 128),
            gnn_layers=gnn_cfg.get("num_layers", 2),
            conv_type=gnn_cfg.get("conv_type", "sage"),
            bert_model_name=bert_cfg.get("model_name", "distilbert-base-uncased"),
            freeze_bert=bert_cfg.get("freeze_encoder", True),
            proj_dim=proj_dim,
            temperature=temperature,
            dropout=gnn_cfg.get("dropout", 0.3),
        )

    def encode_audio(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Extract normalized audio embedding g_i in R^{proj_dim}.
        """
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = getattr(graph_data, "batch", None)
        g_raw, _ = self.audio_encoder(x, edge_index, batch=batch)
        g_proj = self.audio_proj(g_raw)
        return F.normalize(g_proj, p=2, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract normalized text embedding t_i in R^{proj_dim}.
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_cls = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        t_proj = self.text_proj(t_cls)
        return F.normalize(t_proj, p=2, dim=-1)

    def forward(
        self,
        graph_data: Union[Data, Batch],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute normalized audio and text embeddings and symmetric InfoNCE loss.

        Args:
            graph_data: Batched audio graphs (N clips).
            input_ids: Tokenized captions (N captions).
            attention_mask: Attention mask (N, seq_len).

        Returns:
            Dict containing:
                - 'audio_embeds': Normalized audio embeddings (N, proj_dim).
                - 'text_embeds': Normalized text embeddings (N, proj_dim).
                - 'logits_per_audio': Similarity matrix scaled by tau (N, N).
                - 'logits_per_text': Transposed similarity matrix (N, N).
                - 'loss': Symmetric InfoNCE loss scalar.
        """
        audio_embeds = self.encode_audio(graph_data)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # Clamp logit scale to prevent numerical instability
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        # Cosine similarity matrix S_ij = (g_i^T * t_j) / tau
        logits_per_audio = logit_scale * torch.matmul(audio_embeds, text_embeds.T)
        logits_per_text = logits_per_audio.T

        # Ground truth diagonal targets: pair i matches pair i
        batch_size = audio_embeds.size(0)
        labels = torch.arange(batch_size, device=audio_embeds.device)

        # Symmetric InfoNCE loss
        loss_a2t = F.cross_entropy(logits_per_audio, labels)
        loss_t2a = F.cross_entropy(logits_per_text, labels)
        total_loss = 0.5 * (loss_a2t + loss_t2a)

        return {
            "audio_embeds": audio_embeds,
            "text_embeds": text_embeds,
            "logits_per_audio": logits_per_audio,
            "logits_per_text": logits_per_text,
            "loss": total_loss,
        }


# ---------------------------------------------------------------------------
# Cross-Modal Retrieval Evaluation Metrics
# ---------------------------------------------------------------------------
def compute_retrieval_metrics(
    audio_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    top_k: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Evaluate bidirectional retrieval performance:
    1. Text -> Audio: Query caption to retrieve corresponding audio clip.
    2. Audio -> Text: Query audio clip to retrieve corresponding caption.

    Args:
        audio_embeds: Tensor of shape (N, D), L2-normalized.
        text_embeds: Tensor of shape (N, D), L2-normalized.
        top_k: K thresholds for Recall@K evaluation.

    Returns:
        Dictionary of R@1, R@5, R@10 and MRR metrics for both directions.
    """
    N = audio_embeds.size(0)
    # Cosine similarity matrix: (N, N) where entry (i, j) is sim(audio_i, text_j)
    sim_matrix = torch.matmul(audio_embeds, text_embeds.T).detach().cpu().numpy()

    metrics: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Text -> Audio Retrieval (For each caption j, rank all audio candidates i)
    # -----------------------------------------------------------------------
    # Rankings along columns (audio dimension)
    ranks_t2a = []
    for j in range(N):
        # Similarities of all audio tracks to caption j
        scores = sim_matrix[:, j]
        # Rank descending
        sorted_indices = np.argsort(scores)[::-1]
        # Find position of ground-truth audio j
        rank = int(np.where(sorted_indices == j)[0][0]) + 1
        ranks_t2a.append(rank)

    ranks_t2a = np.array(ranks_t2a)
    for k in top_k:
        metrics[f"t2a_R@{k}"] = float(np.mean(ranks_t2a <= k))
    metrics["t2a_MRR"] = float(np.mean(1.0 / ranks_t2a))

    # -----------------------------------------------------------------------
    # Audio -> Text Retrieval (For each audio i, rank all caption candidates j)
    # -----------------------------------------------------------------------
    ranks_a2t = []
    for i in range(N):
        # Similarities of all captions to audio i
        scores = sim_matrix[i, :]
        sorted_indices = np.argsort(scores)[::-1]
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks_a2t.append(rank)

    ranks_a2t = np.array(ranks_a2t)
    for k in top_k:
        metrics[f"a2t_R@{k}"] = float(np.mean(ranks_a2t <= k))
    metrics["a2t_MRR"] = float(np.mean(1.0 / ranks_a2t))

    return metrics


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.graph_builder import MusicGraphDataset, create_graph_dataloader

    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("Initializing MusicContrastiveDualEncoder from config.yaml...")
    model = MusicContrastiveDualEncoder.from_config(
        "config.yaml",
        in_channels=320,
        proj_dim=128,
        temperature=0.07,
    )
    model.eval()

    # Create dummy graph batch (N=4)
    graph_dataset = MusicGraphDataset(split="train", config_path="config.yaml", num_samples=4)
    graph_loader = create_graph_dataloader(graph_dataset, batch_size=4, shuffle=False)
    graph_batch = next(iter(graph_loader))

    # Create dummy tokenized text batch (N=4, L=16)
    dummy_input_ids = torch.randint(1, 1000, (4, 16))
    dummy_attn_mask = torch.ones(4, 16)

    with torch.no_grad():
        outputs = model(
            graph_data=graph_batch,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attn_mask,
        )

    audio_emb = outputs["audio_embeds"]
    text_emb = outputs["text_embeds"]
    logits = outputs["logits_per_audio"]
    loss = outputs["loss"]

    logger.info("Contrastive outputs verified:")
    logger.info("  - Audio embeddings shape: %s (expected: [4, 128])", tuple(audio_emb.shape))
    logger.info("  - Text embeddings shape: %s (expected: [4, 128])", tuple(text_emb.shape))
    logger.info("  - Similarity logits shape: %s (expected: [4, 4])", tuple(logits.shape))
    logger.info("  - Symmetric InfoNCE loss: %.4f", loss.item())

    # Check L2-normalization
    audio_norms = torch.norm(audio_emb, p=2, dim=-1)
    text_norms = torch.norm(text_emb, p=2, dim=-1)
    assert torch.allclose(audio_norms, torch.ones_like(audio_norms), atol=1e-5), "Audio embeds not L2-normalized"
    assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5), "Text embeds not L2-normalized"
    assert logits.shape == (4, 4)
    assert not torch.isnan(loss)

    # Verify retrieval metrics function
    logger.info("Testing retrieval metrics calculation...")
    metrics = compute_retrieval_metrics(audio_emb, text_emb, top_k=(1, 3))
    logger.info("Calculated metrics: %s", metrics)
    assert "t2a_R@1" in metrics
    assert "a2t_R@1" in metrics
    assert "t2a_MRR" in metrics

    logger.info("Task 4 Contrastive Dual-Encoder sanity check PASSED successfully.")
