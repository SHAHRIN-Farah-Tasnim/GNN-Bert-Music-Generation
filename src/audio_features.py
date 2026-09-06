"""
Audio Feature Extraction and Segmentation Pipeline
===================================================
Extracts log-mel spectrogram, chroma, and MFCC features from raw audio signals,
normalizes per track, and segments audio into time windows for graph node construction.

Conforms to CSE425 project specification:
- Resample audio to 22,050 Hz.
- Log-mel spectrogram (128 bins) and Chroma (12 bins).
- Segmentation into fixed windows (e.g., 8s) or beat-synchronous segments.
- Hyperparameters read dynamically from config.yaml.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Logging & Seed Utilities (AGENTS.md Compliance)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audio_features")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility and log it as required by AGENTS.md."""
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
# Audio Feature Extraction Class
# ---------------------------------------------------------------------------
class AudioFeatureExtractor:
    """
    Extracts log-mel spectrograms, chroma, and MFCC features from audio signals.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_chroma: int = 12,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
        segment_seconds: float = 8.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_seconds = segment_seconds

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "AudioFeatureExtractor":
        """Instantiate extractor using hyperparameters from config.yaml."""
        config = load_config(config_path)
        audio_cfg = config.get("audio", {})
        return cls(
            sample_rate=audio_cfg.get("sample_rate", 22050),
            n_mels=audio_cfg.get("n_mels", 128),
            n_chroma=audio_cfg.get("n_chroma", 12),
            segment_seconds=audio_cfg.get("segment_seconds", 8.0),
        )

    def load_audio(
        self,
        audio_path: str,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file, resample to target sample rate, and convert to mono.
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, duration=duration)
        return y, sr

    def compute_log_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Compute normalized log-mel spectrogram (n_mels, time_frames).
        """
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Per-track zero-mean unit-variance normalization
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-8
        norm_log_mel = (log_mel - mean) / std
        return norm_log_mel

    def compute_chroma(self, y: np.ndarray) -> np.ndarray:
        """
        Compute 12-dimensional chroma features (n_chroma, time_frames).
        """
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )
        return chroma

    def compute_mfcc(self, y: np.ndarray) -> np.ndarray:
        """
        Compute MFCC features (n_mfcc, time_frames).
        """
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc

    def segment_audio(
        self,
        y: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Split audio waveform into non-overlapping or sliding segments of length `segment_seconds`.
        """
        segment_len = int(self.sample_rate * self.segment_seconds)
        if len(y) < segment_len:
            # Pad short audio to full segment length
            padded = np.pad(y, (0, segment_len - len(y)), mode="constant")
            return [padded]

        segments = []
        for start in range(0, len(y) - segment_len + 1, segment_len):
            segments.append(y[start : start + segment_len])

        # If leftover tail is significant (>= 50% of segment), pad and include
        remainder = len(y) % segment_len
        if remainder >= segment_len // 2:
            tail = y[-remainder:]
            tail_padded = np.pad(tail, (0, segment_len - remainder), mode="constant")
            segments.append(tail_padded)

        return segments

    def extract_segment_features(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Segment the audio and extract fixed-size summary feature vector for each segment.
        Each segment feature vector h_i^(0) concatenates:
        - Mean & std of log-mel spectrogram across time (2 * n_mels)
        - Mean & std of chroma across time (2 * n_chroma)
        - Mean & std of MFCCs across time (2 * n_mfcc)

        Returns:
            segment_features: numpy array of shape (num_segments, feature_dim).
        """
        segments = self.segment_audio(y)
        feat_list = []

        for seg in segments:
            mel = self.compute_log_mel_spectrogram(seg)
            chroma = self.compute_chroma(seg)
            mfcc = self.compute_mfcc(seg)

            mel_mean, mel_std = np.mean(mel, axis=1), np.std(mel, axis=1)
            chroma_mean, chroma_std = np.mean(chroma, axis=1), np.std(chroma, axis=1)
            mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)

            # Concatenate summary statistics into a single node embedding h_i^(0)
            node_feat = np.concatenate(
                [mel_mean, mel_std, chroma_mean, chroma_std, mfcc_mean, mfcc_std]
            )
            feat_list.append(node_feat)

        return np.stack(feat_list, axis=0)


def generate_synthetic_audio(
    duration: float = 30.0,
    sample_rate: int = 22050,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic multi-frequency audio for testing and validation when
    raw audio datasets are not yet downloaded.
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fundamental harmonic tones (A440 + harmonics) + subtle pink noise
    freqs = [220.0, 440.0, 660.0, 880.0]
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
    noise = np.random.normal(0, 0.05, size=t.shape)
    audio = (signal + noise).astype(np.float32)
    # Normalize amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("Initializing AudioFeatureExtractor from config...")
    extractor = AudioFeatureExtractor.from_config("config.yaml")

    logger.info("Generating synthetic audio clip (30s) for pipeline verification...")
    audio = generate_synthetic_audio(duration=30.0, sample_rate=extractor.sample_rate, seed=seed)

    mel = extractor.compute_log_mel_spectrogram(audio)
    chroma = extractor.compute_chroma(audio)
    mfcc = extractor.compute_mfcc(audio)
    logger.info("Log-mel spectrogram shape: %s", tuple(mel.shape))
    logger.info("Chroma shape: %s", tuple(chroma.shape))
    logger.info("MFCC shape: %s", tuple(mfcc.shape))

    logger.info("Extracting segmented graph node features...")
    node_features = extractor.extract_segment_features(audio)
    logger.info("Segment node features shape: %s", tuple(node_features.shape))

    expected_dim = (2 * extractor.n_mels) + (2 * extractor.n_chroma) + (2 * extractor.n_mfcc)
    assert node_features.shape[1] == expected_dim, (
        f"Feature dimension mismatch: expected {expected_dim}, got {node_features.shape[1]}"
    )
    assert not np.isnan(node_features).any(), "Node features contain NaN values"
    logger.info("AudioFeatureExtractor sanity check PASSED successfully.")
