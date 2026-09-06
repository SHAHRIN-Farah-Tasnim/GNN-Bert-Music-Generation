"""
Task 3: GNN-BERT Fusion for Multi-Context Music Understanding
============================================================
Implements cross-modal fusion combining structural graph embeddings (GNN)
with contextual textual semantics (BERT) for:
1. Multi-label context prediction (genres, mood, instruments).
2. Optional multi-task continuous emotion regression (Valence & Arousal).
3. Comprehensive ablation modes:
   - 'cross_attention': A = softmax(Q K^T / sqrt(d)), z = CONCAT(g, A H_text)
   - 'early_concat': z = CONCAT(g, t_CLS)
   - 'gnn_only': z = g
   - 'bert_only': z = t_CLS

Hyperparameters read from config.yaml as per AGENTS.md rules.
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

from src.bert_encoder import MusicBertClassifier
from src.gnn_model import MusicGNNEncoder

# ---------------------------------------------------------------------------
# Logging & Seed Utilities (AGENTS.md Compliance)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fusion_model")


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
# Cross-Attention Fusion Block
# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion mechanism:
        Q = g * W_Q
        K = H_text * W_K,  V = H_text * W_V
        A = softmax(Q * K^T / sqrt(d))
        c = A * V
        z = CONCAT(g, c)
    """

    def __init__(
        self,
        gnn_dim: int = 128,
        bert_dim: int = 768,
        att_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.bert_dim = bert_dim
        self.att_dim = att_dim
        self.scale = 1.0 / (att_dim**0.5)

        self.w_q = nn.Linear(gnn_dim, att_dim, bias=False)
        self.w_k = nn.Linear(bert_dim, att_dim, bias=False)
        self.w_v = nn.Linear(bert_dim, att_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fused_dim = gnn_dim + att_dim

    def forward(
        self,
        g: torch.Tensor,
        h_text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: Graph readout vectors of shape (batch_size, gnn_dim).
            h_text: Text token representations of shape (batch_size, seq_len, bert_dim).
            attention_mask: Optional mask of shape (batch_size, seq_len) with 1 for valid, 0 for pad.

        Returns:
            Tuple of:
                - z: Fused representation of shape (batch_size, gnn_dim + att_dim).
                - attn_weights: Attention matrix A of shape (batch_size, 1, seq_len).
        """
        # Q: (B, 1, att_dim)
        q = self.w_q(g).unsqueeze(1)
        # K, V: (B, L, att_dim)
        k = self.w_k(h_text)
        v = self.w_v(h_text)

        # Scaled dot-product attention scores: (B, 1, L)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        if attention_mask is not None:
            # Mask out padding tokens with large negative value
            mask = (1.0 - attention_mask.unsqueeze(1).float()) * -1e9
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vector c: (B, 1, att_dim) -> (B, att_dim)
        context = torch.bmm(attn_weights, v).squeeze(1)

        # z = CONCAT(g, c)
        z = torch.cat([g, context], dim=-1)
        return z, attn_weights


# ---------------------------------------------------------------------------
# Task 3 GNN-BERT Fusion Model
# ---------------------------------------------------------------------------
class MusicGNNBertFusionModel(nn.Module):
    """
    Task 3: End-to-end GNN-BERT Fusion Model for multi-context music understanding.
    Supports cross-attention fusion, early concatenation, and single-modality ablations.
    """

    def __init__(
        self,
        in_channels: int = 320,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        conv_type: str = "sage",
        bert_model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = True,
        num_tags: int = 50,
        fusion_mode: str = "cross_attention",
        att_dim: int = 128,
        dropout: float = 0.3,
        alpha_valence: float = 0.5,
        beta_arousal: float = 0.5,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode.lower()
        self.num_tags = num_tags
        self.alpha_valence = alpha_valence
        self.beta_arousal = beta_arousal

        # 1. GNN Audio Branch
        self.gnn_encoder = MusicGNNEncoder(
            in_channels=in_channels,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            conv_type=conv_type,
            dropout=dropout,
        )
        self.gnn_dim = gnn_hidden_dim

        # 2. BERT Text Branch
        logger.info("Initializing BERT encoder in fusion model: %s", bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT parameters frozen in fusion model.")

        # 3. Fusion Layer
        if self.fusion_mode == "cross_attention":
            self.fusion_layer = CrossAttentionFusion(
                gnn_dim=self.gnn_dim,
                bert_dim=self.bert_dim,
                att_dim=att_dim,
                dropout=dropout,
            )
            self.fused_dim = self.fusion_layer.fused_dim
        elif self.fusion_mode == "early_concat":
            self.fusion_layer = None
            self.fused_dim = self.gnn_dim + self.bert_dim
        elif self.fusion_mode == "gnn_only":
            self.fusion_layer = None
            self.fused_dim = self.gnn_dim
        elif self.fusion_mode == "bert_only":
            self.fusion_layer = None
            self.fused_dim = self.bert_dim
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        # 4. Multi-Label Tag Classifier Head
        self.dropout = nn.Dropout(dropout)
        self.tag_classifier = nn.Linear(self.fused_dim, num_tags)
        self.tag_loss_fn = nn.BCEWithLogitsLoss()

        # 5. Multi-Task Continuous Emotion Head (Valence, Arousal)
        self.emotion_head = nn.Linear(self.fused_dim, 2)
        self.emotion_loss_fn = nn.MSELoss()

    @classmethod
    def from_config(
        cls,
        config_path: str = "config.yaml",
        in_channels: int = 320,
        num_tags: int = 50,
        fusion_mode: str = "cross_attention",
    ) -> "MusicGNNBertFusionModel":
        """Instantiate fusion model using hyperparameters from config.yaml."""
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
            num_tags=num_tags,
            fusion_mode=fusion_mode,
            dropout=gnn_cfg.get("dropout", 0.3),
        )

    def forward(
        self,
        graph_data: Union[Data, Batch],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tag_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass combining audio graph and text inputs.

        Args:
            graph_data: Batched PyG Data object.
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            tag_labels: Optional multi-hot tags tensor of shape (batch_size, num_tags).
            emotion_labels: Optional continuous emotion targets (valence, arousal) of shape (batch_size, 2).

        Returns:
            Dictionary containing:
                - 'tag_logits': Tag prediction logits (batch_size, num_tags).
                - 'tag_probs': Tag sigmoid probabilities (batch_size, num_tags).
                - 'emotion_preds': Valence and arousal predictions (batch_size, 2).
                - 'fused_embedding': Combined representation z (batch_size, fused_dim).
                - 'attention_weights': Cross-attention map A (if fusion_mode == 'cross_attention').
                - 'loss': Total multi-task loss (if targets provided).
        """
        # 1. Graph representation g
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = getattr(graph_data, "batch", None)
        g, h_nodes = self.gnn_encoder(x, edge_index, batch=batch)

        # 2. Text representations
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_text = bert_outputs.last_hidden_state  # (B, L, d_bert)
        t_cls = h_text[:, 0, :]  # CLS vector: (B, d_bert)

        # 3. Modality Fusion
        attn_weights = None
        if self.fusion_mode == "cross_attention":
            z, attn_weights = self.fusion_layer(g, h_text, attention_mask=attention_mask)
        elif self.fusion_mode == "early_concat":
            z = torch.cat([g, t_cls], dim=-1)
        elif self.fusion_mode == "gnn_only":
            z = g
        elif self.fusion_mode == "bert_only":
            z = t_cls
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        # 4. Predictions
        z_dropped = self.dropout(z)
        tag_logits = self.tag_classifier(z_dropped)
        tag_probs = torch.sigmoid(tag_logits)
        emotion_preds = self.emotion_head(z_dropped)

        result = {
            "tag_logits": tag_logits,
            "tag_probs": tag_probs,
            "emotion_preds": emotion_preds,
            "fused_embedding": z,
            "graph_embedding": g,
            "cls_embedding": t_cls,
        }
        if attn_weights is not None:
            result["attention_weights"] = attn_weights

        # 5. Multi-task loss computation
        total_loss = 0.0
        has_loss = False

        if tag_labels is not None:
            tag_loss = self.tag_loss_fn(tag_logits, tag_labels.float())
            result["tag_loss"] = tag_loss
            total_loss = total_loss + tag_loss
            has_loss = True

        if emotion_labels is not None:
            # valence is column 0, arousal is column 1
            val_loss = self.emotion_loss_fn(emotion_preds[:, 0], emotion_labels[:, 0].float())
            aro_loss = self.emotion_loss_fn(emotion_preds[:, 1], emotion_labels[:, 1].float())
            emotion_loss = (self.alpha_valence * val_loss) + (self.beta_arousal * aro_loss)
            result["emotion_loss"] = emotion_loss
            total_loss = total_loss + emotion_loss
            has_loss = True

        if has_loss:
            result["loss"] = total_loss

        return result


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.graph_builder import MusicGraphDataset, create_graph_dataloader

    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("Initializing MusicGNNBertFusionModel with cross-attention...")
    model = MusicGNNBertFusionModel.from_config(
        "config.yaml",
        in_channels=320,
        num_tags=50,
        fusion_mode="cross_attention",
    )
    model.eval()

    # Create dummy graph batch (B=2)
    graph_dataset = MusicGraphDataset(split="train", config_path="config.yaml", num_samples=2)
    graph_loader = create_graph_dataloader(graph_dataset, batch_size=2, shuffle=False)
    graph_batch = next(iter(graph_loader))

    # Create dummy tokenized text batch (B=2, L=16)
    dummy_input_ids = torch.randint(1, 1000, (2, 16))
    dummy_attn_mask = torch.ones(2, 16)
    dummy_tags = torch.randint(0, 2, (2, 50)).float()
    dummy_emotion = torch.tensor([[5.2, 6.1], [3.4, 4.0]], dtype=torch.float32)

    with torch.no_grad():
        out = model(
            graph_data=graph_batch,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attn_mask,
            tag_labels=dummy_tags,
            emotion_labels=dummy_emotion,
        )

    logger.info("Fusion outputs verified:")
    logger.info("  - Tag logits shape: %s", tuple(out["tag_logits"].shape))
    logger.info("  - Emotion predictions shape: %s", tuple(out["emotion_preds"].shape))
    logger.info("  - Fused embedding z shape: %s", tuple(out["fused_embedding"].shape))
    logger.info("  - Cross-attention weights shape: %s", tuple(out["attention_weights"].shape))
    logger.info("  - Total multi-task loss: %.4f", out["loss"].item())

    assert out["tag_logits"].shape == (2, 50)
    assert out["emotion_preds"].shape == (2, 2)
    assert out["attention_weights"].shape == (2, 1, 16)
    assert not torch.isnan(out["loss"])

    # Test ablation modes
    logger.info("Testing ablation mode 'early_concat'...")
    model_early = MusicGNNBertFusionModel.from_config(
        "config.yaml", in_channels=320, num_tags=50, fusion_mode="early_concat"
    )
    model_early.eval()
    with torch.no_grad():
        out_early = model_early(graph_batch, dummy_input_ids, dummy_attn_mask)
    expected_early_dim = model.gnn_dim + model.bert_dim
    assert out_early["fused_embedding"].shape == (2, expected_early_dim)
    logger.info("Ablation 'early_concat' verified (dim: %d).", expected_early_dim)

    logger.info("Task 3 Fusion Model and Ablations sanity check PASSED successfully.")
