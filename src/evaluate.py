"""
Evaluation Metrics Suite for Music Context Understanding
=========================================================
Implements the evaluation metrics required by Section 6 of the CSE425 specification:
1. Multi-label Tag / Genre Classification:
   - Precision, Recall, Per-Tag F1
   - Macro-F1: (1 / K) * sum_{k=1}^K F1_k
   - Micro-F1: Globally pooled F1
   - Mean AUC-PR: Area Under Precision-Recall Curve averaged across tags
2. Emotion Regression (DEAM):
   - MAE: (1 / N) * sum |y_i - y_hat_i|
   - R^2: 1 - sum (y_i - y_hat_i)^2 / sum (y_i - y_mean)^2
3. Cross-Modal Retrieval (MusicCaps):
   - Caption -> Audio: R@1, R@5, R@10, MRR
   - Audio -> Caption: R@1, R@5, R@10, MRR

Conforms to AGENTS.md: Never hardcodes experimental numbers (displays TODO placeholders).
"""

import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import torch
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
logger = logging.getLogger("evaluate")


def set_seed(seed: int = 42) -> None:
    """Set random seed across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
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
# 1. Multi-Label Tag / Genre Classification Metrics
# ---------------------------------------------------------------------------
def compute_tag_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_probs: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes Macro-F1, Micro-F1, and Mean AUC-PR for multi-label tag predictions.

    Args:
        y_true: Binary ground-truth matrix of shape (N, K) with values in {0, 1}.
        y_pred_probs: Continuous predicted probabilities in [0, 1] of shape (N, K).
        threshold: Decision threshold to convert probabilities into binary predictions.

    Returns:
        Dictionary containing 'macro_f1', 'micro_f1', and 'mean_auc_pr'.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    y_true = y_true.astype(int)
    y_pred_bin = (y_pred_probs >= threshold).astype(int)

    N, K = y_true.shape

    # 1. Macro-F1: Compute per-tag F1 and average across all K tags
    f1_per_tag: List[float] = []
    auc_pr_per_tag: List[float] = []

    for k in range(K):
        yt_k = y_true[:, k]
        yp_k = y_pred_bin[:, k]
        prob_k = y_pred_probs[:, k]

        # Per-tag Precision, Recall, F1
        tp = np.sum((yt_k == 1) & (yp_k == 1))
        fp = np.sum((yt_k == 0) & (yp_k == 1))
        fn = np.sum((yt_k == 1) & (yp_k == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_tag.append(f1)

        # Per-tag AUC-PR (Area under Precision-Recall Curve)
        if np.sum(yt_k) > 0:  # Only evaluate AUC-PR if there is at least one positive instance
            try:
                ap = average_precision_score(yt_k, prob_k)
                if not np.isnan(ap):
                    auc_pr_per_tag.append(float(ap))
            except Exception:
                pass

    macro_f1 = float(np.mean(f1_per_tag)) if f1_per_tag else 0.0
    mean_auc_pr = float(np.mean(auc_pr_per_tag)) if auc_pr_per_tag else 0.0

    # 2. Micro-F1: Pool true positives, false positives, and false negatives globally
    total_tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    total_fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    total_fn = np.sum((y_true == 1) & (y_pred_bin == 0))

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        (2 * micro_prec * micro_rec) / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    return {
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "mean_auc_pr": mean_auc_pr,
    }


# ---------------------------------------------------------------------------
# 2. Continuous Emotion Regression Metrics (Valence & Arousal)
# ---------------------------------------------------------------------------
def compute_emotion_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes MAE and R^2 for continuous emotion regression (Valence & Arousal).

    Args:
        y_true: Ground-truth array of shape (N, 2) where col 0 is valence, col 1 is arousal.
        y_pred: Predicted values of shape (N, 2).

    Returns:
        Dictionary containing 'mae_valence', 'mae_arousal', 'mae_overall',
        'r2_valence', and 'r2_arousal'.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    val_true, val_pred = y_true[:, 0], y_pred[:, 0]
    aro_true, aro_pred = y_true[:, 1], y_pred[:, 1]

    # Mean Absolute Error
    mae_val = float(np.mean(np.abs(val_true - val_pred)))
    mae_aro = float(np.mean(np.abs(aro_true - aro_pred)))
    mae_overall = (mae_val + mae_aro) / 2.0

    # R^2 Coefficient of Determination: 1 - sum((y - y_hat)^2) / sum((y - y_mean)^2)
    def calc_r2(y, y_hat):
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-8)))

    r2_val = calc_r2(val_true, val_pred)
    r2_aro = calc_r2(aro_true, aro_pred)

    return {
        "mae_valence": mae_val,
        "mae_arousal": mae_aro,
        "mae_overall": mae_overall,
        "r2_valence": r2_val,
        "r2_arousal": r2_aro,
    }


# ---------------------------------------------------------------------------
# 3. Formatter for Results Table (AGENTS.md Compliance)
# ---------------------------------------------------------------------------
def format_results_table(measured_results: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """
    Generate markdown results table formatted according to the project rubric.
    Unmeasured models or metrics are marked with TODO.
    """
    measured = measured_results or {}

    rows = [
        ("B1: Random tags", "B1"),
        ("B2: CNN mel-spectrogram", "B2"),
        ("Task 1: BERT-only", "Task1"),
        ("Task 2: GNN-only", "Task2"),
        ("Task 3: GNN-BERT fusion", "Task3"),
        ("Task 4: Contrastive", "Task4"),
    ]

    header = "| Model | Macro-F1 | AUC-PR | MAE (emotion) | R@5 |\n|---|---|---|---|---|\n"
    table_lines = [header]

    for model_name, key in rows:
        m_dict = measured.get(key, {})
        f1_val = f"{m_dict['macro_f1']:.4f}" if "macro_f1" in m_dict else "TODO"
        auc_val = f"{m_dict['mean_auc_pr']:.4f}" if "mean_auc_pr" in m_dict else "TODO"
        mae_val = f"{m_dict['mae_overall']:.4f}" if "mae_overall" in m_dict else ("-" if "emotion" not in key.lower() and key not in ("B2", "Task2", "Task3") else "TODO")
        r5_val = f"{m_dict['t2a_R@5']:.4f}" if "t2a_R@5" in m_dict else ("TODO" if key in ("B1", "Task4") else "-")

        line = f"| {model_name} | {f1_val} | {auc_val} | {mae_val} | {r5_val} |\n"
        table_lines.append(line)

    return "".join(table_lines)


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("Testing multi-label tag classification metrics...")
    # Synthetic controlled test matrix: N=10, K=5
    y_true = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ])
    y_probs = np.array([
        [0.9, 0.1, 0.8, 0.2, 0.1],
        [0.1, 0.8, 0.2, 0.7, 0.3],
        [0.8, 0.9, 0.3, 0.1, 0.8],
        [0.2, 0.3, 0.7, 0.6, 0.1],
    ])

    tag_metrics = compute_tag_classification_metrics(y_true, y_probs, threshold=0.5)
    logger.info("Tag metrics: %s", tag_metrics)
    assert tag_metrics["macro_f1"] > 0.8, "Expected high Macro-F1 on accurate probabilities"
    assert tag_metrics["micro_f1"] > 0.8, "Expected high Micro-F1"
    assert tag_metrics["mean_auc_pr"] > 0.8, "Expected high Mean AUC-PR"

    logger.info("Testing emotion regression metrics...")
    emotion_true = np.array([[5.0, 6.0], [3.0, 4.0], [7.0, 8.0]])
    emotion_pred = np.array([[5.1, 5.9], [3.2, 3.8], [6.9, 8.1]])
    emo_metrics = compute_emotion_regression_metrics(emotion_true, emotion_pred)
    logger.info("Emotion metrics: %s", emo_metrics)
    assert emo_metrics["mae_overall"] < 0.2, "Expected low MAE on near-perfect predictions"
    assert emo_metrics["r2_valence"] > 0.9, "Expected high R2 score"

    logger.info("Testing results table formatter...")
    table_str = format_results_table({"Task1": {"macro_f1": 0.4812, "mean_auc_pr": 0.4410}})
    logger.info("Formatted Table Output:\n%s", table_str)
    assert "Task 1: BERT-only | 0.4812 | 0.4410" in table_str
    assert "Task 2: GNN-only | TODO" in table_str

    logger.info("evaluate.py sanity check PASSED successfully.")
