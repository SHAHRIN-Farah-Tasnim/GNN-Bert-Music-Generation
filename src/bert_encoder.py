"""
Task 1: BERT Baseline for Music Tag Understanding
=================================================
Implements a BERT-based multi-label tag classifier using HuggingFace
`distilbert-base-uncased` with a linear classification head on the CLS token
and BCEWithLogitsLoss, in accordance with the CSE425 project specification.

Mathematical Formulation:
    t = BERT_CLS(X_text)
    y_hat_k = sigma(w_k^T * t + b_k)
    L_BERT = - (1 / K) * sum_{k=1}^K [y_k * log(y_hat_k) + (1 - y_k) * log(1 - y_hat_k)]

All hyperparameters are read from config.yaml as per AGENTS.md rules.
Metrics and evaluation numbers are not hardcoded (marked as TODO pending experimental runs).
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import yaml

# ---------------------------------------------------------------------------
# Logging & Seed Utilities (AGENTS.md Compliance)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bert_encoder")


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across random, numpy, and PyTorch for reproducibility.
    Logs the seed as required by AGENTS.md.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to: %d", seed)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from config.yaml.
    """
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
# Task 1 Model: DistilBERT Multi-Label Tag Classifier
# ---------------------------------------------------------------------------
class MusicBertClassifier(nn.Module):
    """
    BERT-based multi-label classifier for textual music context (Task 1).

    Uses distilbert-base-uncased backbone from HuggingFace, extracts the CLS
    token representation, and applies a linear classification head with
    BCEWithLogitsLoss for multi-label tag prediction. Also exposes token-level
    hidden states H_text and CLS vector t for downstream GNN-BERT fusion (Task 3)
    and cross-modal contrastive learning (Task 4).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 50,
        freeze_encoder: bool = False,
        dropout: float = 0.2,
    ):
        """
        Args:
            model_name: Pretrained HuggingFace model identifier.
            num_labels: Number of target tag classes (e.g., top-50 MagnaTagATune tags).
            freeze_encoder: Whether to freeze the pretrained transformer parameters.
            dropout: Dropout probability before the linear classification head.
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.freeze_encoder = freeze_encoder

        # Load pretrained transformer backbone
        logger.info("Loading pretrained BERT backbone: %s", model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size

        # Freeze encoder parameters if specified in config.yaml
        if freeze_encoder:
            logger.info("Freezing BERT encoder parameters")
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            logger.info("BERT encoder parameters are trainable (fine-tuning enabled)")

        # Linear classification head on top of the CLS token
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, num_labels)

        # Multi-label loss: Binary Cross Entropy with Logits (per-tag binary cross-entropy)
        self.loss_fn = nn.BCEWithLogitsLoss()

    @classmethod
    def from_config(cls, config_path: str = "config.yaml", num_labels: int = 50) -> "MusicBertClassifier":
        """
        Instantiate MusicBertClassifier directly from config.yaml parameters.
        """
        config = load_config(config_path)
        bert_cfg = config.get("bert", {})
        model_name = bert_cfg.get("model_name", "distilbert-base-uncased")
        freeze_encoder = bert_cfg.get("freeze_encoder", True)

        return cls(
            model_name=model_name,
            num_labels=num_labels,
            freeze_encoder=freeze_encoder,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multi-label tag classifier.

        Args:
            input_ids: Tokenized input IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            labels: Optional ground-truth multi-hot binary labels of shape (batch_size, num_labels).

        Returns:
            Dict containing:
                - 'logits': Raw linear head outputs of shape (batch_size, num_labels).
                - 'probabilities': Sigmoid probabilities of shape (batch_size, num_labels).
                - 'cls_embedding': CLS vector t of shape (batch_size, hidden_dim).
                - 'hidden_states': Full contextual sequence H_text of shape (batch_size, seq_len, hidden_dim).
                - 'loss': BCEWithLogitsLoss scalar (if labels are provided).
        """
        # Pass through DistilBERT backbone
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.last_hidden_state

        # Extract CLS token representation t (position 0)
        cls_embedding = hidden_states[:, 0, :]  # shape: (batch_size, hidden_dim)

        # Classification head: y_hat = W * t + b
        pooled = self.dropout(cls_embedding)
        logits = self.classifier(pooled)  # shape: (batch_size, num_labels)
        probabilities = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "cls_embedding": cls_embedding,
            "hidden_states": hidden_states,
        }

        # Calculate BCEWithLogitsLoss if labels are supplied
        if labels is not None:
            # Ensure float tensor for BCEWithLogitsLoss
            loss = self.loss_fn(logits, labels.float())
            result["loss"] = loss

        return result

    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract CLS token embedding t = BERT_CLS(X_text) for Task 3 and Task 4.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def get_token_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract token-level representations H_text in R^{L x d} for cross-attention fusion (Task 3).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


# Alias for flexible naming across modules
BertTagClassifier = MusicBertClassifier


# ---------------------------------------------------------------------------
# Task 1 Dataset: Music Tag Dataset Skeleton
# ---------------------------------------------------------------------------
class MusicTagDataset(Dataset):
    """
    Dataset class for textual music context (tags, lyrics, or captions) and multi-label targets.

    The dataset is not downloaded yet — this class provides the full tokenization,
    collation, and label handling skeleton with clear TODO placeholders where the
    data loading logic will reside once the dataset files are placed in data/raw.
    """

    def __init__(
        self,
        split: str = "train",
        config_path: str = "config.yaml",
        tokenizer: Optional[Any] = None,
        max_length: Optional[int] = None,
        num_labels: int = 50,
        texts: Optional[List[str]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        use_dummy_if_missing: bool = True,
    ):
        """
        Args:
            split: One of 'train', 'val', 'test'.
            config_path: Path to config.yaml.
            tokenizer: Pretrained HuggingFace tokenizer instance (auto-initialized if None).
            max_length: Maximum sequence length (defaults to bert.max_length in config.yaml).
            num_labels: Number of multi-label tags.
            texts: Optional explicit list of raw strings.
            labels: Optional explicit multi-hot labels array (N, num_labels).
            use_dummy_if_missing: If True and dataset files are not found, initializes dummy
                                  samples so that code verification and model sanity checks pass.
        """
        self.split = split
        self.config = load_config(config_path)
        self.num_labels = num_labels

        # Resolve sequence length from config if not explicitly passed
        if max_length is None:
            self.max_length = self.config.get("bert", {}).get("max_length", 128)
        else:
            self.max_length = max_length

        # Initialize tokenizer if not provided
        if tokenizer is None:
            model_name = self.config.get("bert", {}).get("model_name", "distilbert-base-uncased")
            logger.info("Initializing tokenizer for: %s", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        # -------------------------------------------------------------------
        # DATA LOADING LOGIC (TODO: Connect to downloaded dataset)
        # -------------------------------------------------------------------
        if texts is not None and labels is not None:
            self.texts = list(texts)
            self.labels = np.array(labels, dtype=np.float32)
        else:
            self.texts, self.labels = self._load_data(use_dummy_if_missing)

    def _load_data(self, use_dummy_if_missing: bool) -> Tuple[List[str], np.ndarray]:
        """
        Load dataset split from disk, or fallback to dummy samples if not downloaded yet.
        """
        raw_path = self.config.get("paths", {}).get("raw", "data/raw")
        splits_path = self.config.get("paths", {}).get("splits", "data/splits")

        # ===================================================================
        # TODO: DATASET LOADING IMPLEMENTATION
        # When dataset (MagnaTagATune, MusicCaps, or FMA) is downloaded:
        #
        # 1. MagnaTagATune (top-50 tags):
        #    - Load annotations file: os.path.join(raw_path, "annotations_final.csv")
        #    - Read official train/val/test split indices from splits_path.
        #    - Identify top-50 most frequent tag columns.
        #    - Create textual context X_text for each clip (e.g., concatenated tags,
        #      available audio caption/description, or artist/title metadata).
        #    - Extract multi-hot binary label vector y in {0, 1}^50.
        #
        # 2. MusicCaps (captions -> tags):
        #    - Load os.path.join(raw_path, "musiccaps-public.csv").
        #    - Use natural language caption column 'caption' as X_text.
        #    - Parse 'aspect_list' into multi-hot tag vector y.
        #
        # 3. Assign:
        #    - texts = [str(x) for x in df_split["text_context"]]
        #    - labels = df_split[tag_cols].to_numpy(dtype=np.float32)
        # ===================================================================

        dataset_ready = False
        mtt_annotations = os.path.join(raw_path, "annotations_final.csv")
        musiccaps_file = os.path.join(raw_path, "musiccaps-public.csv")

        if os.path.exists(mtt_annotations):
            logger.info("Found MagnaTagATune annotations at: %s", mtt_annotations)
            # TODO: Add pandas parsing and split filtering logic here once downloaded.
            dataset_ready = True
        elif os.path.exists(musiccaps_file):
            logger.info("Found MusicCaps file at: %s", musiccaps_file)
            # TODO: Add MusicCaps parsing logic here once downloaded.
            dataset_ready = True

        if not dataset_ready:
            if use_dummy_if_missing:
                logger.warning(
                    "Dataset not downloaded yet (looked in '%s'). "
                    "Populating %s split with synthetic dummy samples for Task 1 pipeline verification.",
                    raw_path,
                    self.split,
                )
                dummy_texts = [
                    "melancholic guitar with slow rock tempo and electric solo",
                    "fast electronic dance beat with heavy synthesizer bass",
                    "calm acoustic piano and classical violin melody",
                    "ambient instrumental music with atmospheric pads",
                    "energetic jazz trumpet solo with walking bass line and drums",
                    "heavy metal distorted electric guitars with aggressive drums",
                    "mellow folk song with female vocals and acoustic guitar",
                    "upbeat pop synth melody with cheerful rhythmic percussion",
                ]
                # Set seed for reproducible dummy labels
                np.random.seed(self.config.get("seed", 42))
                dummy_labels = (np.random.rand(len(dummy_texts), self.num_labels) > 0.85).astype(np.float32)
                return dummy_texts, dummy_labels
            else:
                raise FileNotFoundError(
                    f"Dataset is not downloaded yet. Please place dataset files in '{raw_path}' "
                    f"or set use_dummy_if_missing=True for testing."
                )

        # Fallback return (to be replaced when loading code is executed)
        return [], np.zeros((0, self.num_labels), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
            "text": text,
        }


def create_dataloader(
    dataset: MusicTagDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Utility to create a PyTorch DataLoader for MusicTagDataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load config and set random seed as required by AGENTS.md
    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    bert_cfg = config.get("bert", {})
    model_name = bert_cfg.get("model_name", "distilbert-base-uncased")
    max_len = bert_cfg.get("max_length", 128)
    freeze_enc = bert_cfg.get("freeze_encoder", True)
    batch_size = config.get("train", {}).get("batch_size", 4)
    num_labels = 50  # Top-50 tags per specification

    logger.info("Initializing MusicBertClassifier from config...")
    model = MusicBertClassifier(
        model_name=model_name,
        num_labels=num_labels,
        freeze_encoder=freeze_enc,
    )
    model.eval()

    logger.info("Initializing MusicTagDataset skeleton...")
    dataset = MusicTagDataset(
        split="train",
        config_path="config.yaml",
        num_labels=num_labels,
        use_dummy_if_missing=True,
    )
    dataloader = create_dataloader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    logger.info("Input IDs shape: %s", tuple(input_ids.shape))
    logger.info("Attention mask shape: %s", tuple(attention_mask.shape))
    logger.info("Labels shape: %s", tuple(labels.shape))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    logits = outputs["logits"]
    probs = outputs["probabilities"]
    cls_emb = outputs["cls_embedding"]
    hidden_states = outputs["hidden_states"]
    loss = outputs["loss"]

    logger.info("Outputs verified:")
    logger.info("  - Logits shape: %s (expected: [%d, %d])", tuple(logits.shape), len(input_ids), num_labels)
    logger.info("  - Probabilities shape: %s", tuple(probs.shape))
    logger.info("  - CLS embedding shape: %s (expected: [%d, %d])", tuple(cls_emb.shape), len(input_ids), model.hidden_dim)
    logger.info("  - Hidden states shape: %s (expected: [%d, %d, %d])", tuple(hidden_states.shape), len(input_ids), max_len, model.hidden_dim)
    logger.info("  - BCEWithLogitsLoss: %.4f", loss.item())

    # Verify BCEWithLogitsLoss numerical range
    assert logits.shape == (len(input_ids), num_labels), "Logits shape mismatch"
    assert cls_emb.shape == (len(input_ids), model.hidden_dim), "CLS shape mismatch"
    assert not torch.isnan(loss), "Loss computed to NaN"
    logger.info("Task 1 Model and Dataset sanity check PASSED successfully.")
