"""
Audio Feature Extraction, Segmentation & Chord Analysis Pipeline
================================================================
Extracts log-mel spectrogram, chroma, and MFCC features from raw audio signals,
normalizes per track (preserving inter-segment dynamic contrasts per Spec 3.1),
balances feature scaling, and provides:
1. Fixed sliding windows (with configurable overlap) or beat-synchronous segments (3.2).
2. Segment summary node embeddings with balanced feature standardization.
3. Chord estimation via pitch-class profile template matching for chord-transition graphs (3.3).

Hyperparameters read dynamically from config.yaml.
"""

import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
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
# Chord Templates (24 Triads: 12 Major, 12 Minor) for Spec 3.3
# ---------------------------------------------------------------------------
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def create_chord_templates() -> Tuple[np.ndarray, List[str]]:
    """
    Construct standardized 12-dimensional binary pitch-class profiles (templates)
    for the 24 canonical triads: 12 Major (root, +4, +7) and 12 Minor (root, +3, +7).

    Returns:
        templates: Array of shape (24, 12), L2-normalized.
        chord_names: List of 24 chord names (e.g. 'C:maj', 'A:min').
    """
    templates = []
    chord_names = []

    # 12 Major chords
    for root in range(12):
        prof = np.zeros(12, dtype=np.float32)
        prof[root] = 1.0
        prof[(root + 4) % 12] = 1.0  # Major third
        prof[(root + 7) % 12] = 1.0  # Fifth
        prof = prof / np.linalg.norm(prof)
        templates.append(prof)
        chord_names.append(f"{PITCH_NAMES[root]}:maj")

    # 12 Minor chords
    for root in range(12):
        prof = np.zeros(12, dtype=np.float32)
        prof[root] = 1.0
        prof[(root + 3) % 12] = 1.0  # Minor third
        prof[(root + 7) % 12] = 1.0  # Fifth
        prof = prof / np.linalg.norm(prof)
        templates.append(prof)
        chord_names.append(f"{PITCH_NAMES[root]}:min")

    return np.array(templates, dtype=np.float32), chord_names


CHORD_TEMPLATES, CHORD_NAMES = create_chord_templates()

# Node embedding dimensionality: (2*n_mels) + (2*n_chroma) + (2*n_mfcc)
# with defaults 128/12/20 this is 320. Both the windowed and the
# beat-synchronous segmentation paths MUST emit this same width, otherwise
# gnn_model's in_dim silently breaks when switching segmentation mode.
NODE_FEATURE_DIM = (2 * 128) + (2 * 12) + (2 * 20)


# ---------------------------------------------------------------------------
# Audio Feature Extraction Class
# ---------------------------------------------------------------------------
class AudioFeatureExtractor:
    """
    Extracts per-track normalized log-mel spectrograms, chroma, and MFCC features.
    Provides windowed, overlapping, and beat-synchronous segmentation per Spec 3.1 & 3.2.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_chroma: int = 12,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
        segment_seconds: float = 5.0,
        hop_seconds: Optional[float] = None,
        chord_seconds: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_seconds = segment_seconds
        # 50% window overlap by default to produce richer graphs (10-15 nodes on 30s clips)
        self.hop_seconds = hop_seconds if hop_seconds is not None else (segment_seconds / 2.0)
        # Chord pooling window: real chords last 1-2s, an STFT frame lasts ~23ms
        self.chord_seconds = chord_seconds

    @property
    def node_feature_dim(self) -> int:
        """Width of the node embedding produced by both segmentation paths."""
        return (2 * self.n_mels) + (2 * self.n_chroma) + (2 * self.n_mfcc)

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "AudioFeatureExtractor":
        """
        Instantiate extractor using hyperparameters from config.yaml.

        Every constructor argument is read here. If a parameter is added to the
        constructor but not to this method, setting it in config.yaml silently
        does nothing, which quietly breaks reproducibility.
        """
        config = load_config(config_path)
        audio_cfg = config.get("audio", {})
        seg_sec = audio_cfg.get("segment_seconds", 5.0)
        return cls(
            sample_rate=audio_cfg.get("sample_rate", 22050),
            n_mels=audio_cfg.get("n_mels", 128),
            n_chroma=audio_cfg.get("n_chroma", 12),
            n_mfcc=audio_cfg.get("n_mfcc", 20),
            n_fft=audio_cfg.get("n_fft", 2048),
            hop_length=audio_cfg.get("hop_length", 512),
            segment_seconds=seg_sec,
            hop_seconds=audio_cfg.get("hop_seconds", seg_sec / 2.0),
            chord_seconds=audio_cfg.get("chord_seconds", 1.0),
        )

    def load_audio(
        self,
        audio_path: str,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load an audio file, resample to target sample rate, and convert to mono."""
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, duration=duration)
        return y, sr

    def compute_track_features(
        self,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Extract and per-track normalize log-mel, chroma, and MFCC representations (Spec 3.1).

        Normalization happens ONCE over the full track, not per segment. Normalizing
        each segment independently would force every segment to zero mean and unit
        variance, erasing the loudness and timbre differences BETWEEN segments, which
        is precisely the structure the segment-similarity graph is meant to capture.

        All three feature families are standardized separately so that no single
        family dominates downstream GNN message passing. Without this, raw MFCC
        coefficient 0 (often in the hundreds) swamps the normalized mel dimensions.
        """
        # 1. Log-Mel Spectrogram (128 bins)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel_norm = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

        # 2. Chroma Features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )
        chroma_norm = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-8)

        # 3. MFCC Features (20 coefficients)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        return {
            "log_mel": log_mel_norm,
            "chroma": chroma_norm,
            "mfcc": mfcc_norm,
        }

    @staticmethod
    def _summarize(
        seg_mel: np.ndarray,
        seg_chroma: np.ndarray,
        seg_mfcc: np.ndarray,
    ) -> np.ndarray:
        """
        Build one node embedding from a slice of the three feature families.

        Block order is mel_mean, mel_std, chroma_mean, chroma_std, mfcc_mean,
        mfcc_std. The beat-synchronous path reproduces this exact order so the
        two segmentation modes are interchangeable.
        """
        return np.concatenate(
            [
                np.mean(seg_mel, axis=1),
                np.std(seg_mel, axis=1),
                np.mean(seg_chroma, axis=1),
                np.std(seg_chroma, axis=1),
                np.mean(seg_mfcc, axis=1),
                np.std(seg_mfcc, axis=1),
            ]
        )

    def segment_track_features(
        self,
        track_features: Dict[str, np.ndarray],
        total_duration_sec: float,
    ) -> np.ndarray:
        """
        Slice the per-track normalized features into overlapping time windows and
        compute summary statistics (mean + std) per segment to produce node
        vectors h_i^(0).

        Returns:
            segment_features: Array of shape (num_segments, node_feature_dim).
        """
        log_mel = track_features["log_mel"]
        chroma = track_features["chroma"]
        mfcc = track_features["mfcc"]

        total_frames = log_mel.shape[1]
        frames_per_sec = total_frames / max(total_duration_sec, 1e-6)

        window_frames = max(1, int(self.segment_seconds * frames_per_sec))
        hop_frames = max(1, int(self.hop_seconds * frames_per_sec))

        feat_list = []
        start_frame = 0

        while start_frame + (window_frames // 2) < total_frames:
            end_frame = min(start_frame + window_frames, total_frames)

            feat_list.append(
                self._summarize(
                    log_mel[:, start_frame:end_frame],
                    chroma[:, start_frame:end_frame],
                    mfcc[:, start_frame:end_frame],
                )
            )

            if end_frame >= total_frames:
                break
            start_frame += hop_frames

        if not feat_list:
            # Fallback if audio is extremely short
            feat_list.append(self._summarize(log_mel, chroma, mfcc))

        return np.stack(feat_list, axis=0)

    def segment_audio_beat_sync(
        self,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Beat-synchronous segmentation using librosa.beat.beat_track per Spec 3.2.

        Emits mean AND std per beat, in the same block order as
        segment_track_features, so both segmentation modes produce
        node_feature_dim-wide vectors. Returning a narrower vector here would
        silently break gnn_model's fixed in_dim when switching modes.
        """
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        track_feats = self.compute_track_features(y)
        log_mel = track_feats["log_mel"]
        chroma = track_feats["chroma"]
        mfcc = track_feats["mfcc"]

        if len(beat_frames) > 1:
            beat_feats = np.concatenate(
                [
                    librosa.util.sync(log_mel, beat_frames, aggregate=np.mean),
                    librosa.util.sync(log_mel, beat_frames, aggregate=np.std),
                    librosa.util.sync(chroma, beat_frames, aggregate=np.mean),
                    librosa.util.sync(chroma, beat_frames, aggregate=np.std),
                    librosa.util.sync(mfcc, beat_frames, aggregate=np.mean),
                    librosa.util.sync(mfcc, beat_frames, aggregate=np.std),
                ],
                axis=0,
            ).T
        else:
            # No reliable beat grid detected; fall back to fixed windows
            beat_feats = self.extract_segment_features(y)

        return beat_feats, np.asarray(beat_frames)

    def extract_segment_features(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience end-to-end method: computes per-track features and slices
        into balanced, overlapping segment node embeddings h_i^(0).
        """
        duration = len(y) / self.sample_rate
        track_feats = self.compute_track_features(y)
        return self.segment_track_features(track_feats, duration)

    def extract_chord_sequence(
        self,
        y: np.ndarray,
        chord_seconds: Optional[float] = None,
        collapse_repeats: bool = True,
    ) -> Tuple[List[int], List[str], np.ndarray]:
        """
        Estimate a chord sequence via template matching on chroma (Spec 3.3).

        Chroma is pooled over `chord_seconds` blocks BEFORE template matching.
        A raw STFT frame spans roughly 23 ms while real chords last 1-2 seconds,
        so matching per frame yields on the order of a thousand transitions per
        track, ~70% of which are self-loops. That makes the chord-transition
        graph nearly complete and causes edge weights to encode frame duration
        rather than harmonic movement.

        Consecutive duplicates are then collapsed, so C C C G G becomes C G and
        only genuine chord changes survive as graph edges.

        chroma_cqt is used here rather than chroma_stft: constant-Q bins are
        log-spaced and aligned to musical pitch, making it substantially more
        reliable for harmonic analysis.

        Returns:
            chord_indices: chord index [0..23] per chord span.
            chord_labels: chord names, e.g. ['C:maj', 'G:maj', ...].
            transition_matrix: observed transition counts, shape (24, 24).
        """
        if chord_seconds is None:
            chord_seconds = self.chord_seconds

        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )  # (12, T)

        # Pool chroma into chord-length blocks
        if chord_seconds and chord_seconds > 0:
            frames_per_chord = max(
                1, int(round(chord_seconds * self.sample_rate / self.hop_length))
            )
            pooled = [
                chroma[:, i : i + frames_per_chord].mean(axis=1)
                for i in range(0, chroma.shape[1], frames_per_chord)
            ]
            chroma_blocks = np.stack(pooled, axis=1)
        else:
            chroma_blocks = chroma

        # L2-normalize each block, then correlate against the 24 triad templates
        norms = np.linalg.norm(chroma_blocks, axis=0, keepdims=True) + 1e-8
        correlations = np.dot(CHORD_TEMPLATES, chroma_blocks / norms)
        chord_indices = np.argmax(correlations, axis=0).tolist()

        if collapse_repeats:
            chord_indices = [
                c
                for i, c in enumerate(chord_indices)
                if i == 0 or c != chord_indices[i - 1]
            ]

        chord_labels = [CHORD_NAMES[idx] for idx in chord_indices]

        # Empirical transition count matrix (24 x 24)
        transition_matrix = np.zeros((24, 24), dtype=np.float32)
        for c_curr, c_next in zip(chord_indices[:-1], chord_indices[1:]):
            transition_matrix[c_curr, c_next] += 1.0

        return chord_indices, chord_labels, transition_matrix


def generate_synthetic_audio(
    duration: float = 30.0,
    sample_rate: int = 22050,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic harmonic audio for pipeline validation only.

    This is four sine waves plus noise. It verifies that the pipeline runs
    without crashing and produces correctly shaped output. It says nothing
    about whether the features are musically meaningful, and beat tracking on
    it detects nothing real. Validate on actual audio before drawing any
    conclusion from these features.
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freqs = [220.0, 440.0, 660.0, 880.0]
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
    noise = np.random.normal(0, 0.05, size=t.shape)
    audio = (signal + noise).astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio


# ---------------------------------------------------------------------------
# Self-Verification / Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("Initializing AudioFeatureExtractor from config.yaml...")
    extractor = AudioFeatureExtractor.from_config("config.yaml")
    logger.info(
        "segment_seconds=%.1f  hop_seconds=%.1f  chord_seconds=%.1f  node_dim=%d",
        extractor.segment_seconds,
        extractor.hop_seconds,
        extractor.chord_seconds,
        extractor.node_feature_dim,
    )

    audio = generate_synthetic_audio(duration=30.0, sample_rate=extractor.sample_rate, seed=seed)

    logger.info("Extracting per-track balanced segment features...")
    node_features = extractor.extract_segment_features(audio)
    logger.info(
        "Extracted %d segments with feature shape %s",
        len(node_features),
        tuple(node_features.shape),
    )
    assert node_features.shape[0] >= 6, (
        f"Expected >= 6 nodes on 30s audio, got {node_features.shape[0]}"
    )
    assert node_features.shape[1] == extractor.node_feature_dim, (
        f"Expected {extractor.node_feature_dim} feature dims, got {node_features.shape[1]}"
    )
    assert not np.isnan(node_features).any(), "Node features contain NaN values"

    # Verify the three feature blocks are on comparable scales. If one block
    # dominates numerically it will dominate GNN message passing regardless of
    # how informative it actually is.
    n_mel_block = 2 * extractor.n_mels
    n_chroma_block = 2 * extractor.n_chroma
    mel_std = node_features[:, :n_mel_block].std()
    chroma_std = node_features[:, n_mel_block : n_mel_block + n_chroma_block].std()
    mfcc_std = node_features[:, n_mel_block + n_chroma_block :].std()
    logger.info(
        "Feature block spread -> mel: %.3f  chroma: %.3f  mfcc: %.3f",
        mel_std,
        chroma_std,
        mfcc_std,
    )
    assert max(mel_std, chroma_std, mfcc_std) / (min(mel_std, chroma_std, mfcc_std) + 1e-8) < 20.0, (
        "Feature blocks are on wildly different scales; one will dominate message passing"
    )

    logger.info("Testing chord extraction and transition matrix (3.3)...")
    chord_idx, chord_names_seq, trans_matrix = extractor.extract_chord_sequence(audio)
    n_transitions = int(np.sum(trans_matrix))
    n_self_loops = int(np.trace(trans_matrix))
    logger.info(
        "Chord spans: %d, transitions: %d, unique chords: %d, self-loops: %d",
        len(chord_idx),
        n_transitions,
        len(set(chord_idx)),
        n_self_loops,
    )
    logger.info("First chords: %s", chord_names_seq[:8])
    assert trans_matrix.shape == (24, 24)
    assert n_self_loops == 0, (
        "Collapsing failed: self-loops present, chord graph will be degenerate"
    )
    assert n_transitions < 200, (
        f"Too many transitions ({n_transitions}); chord pooling is not working"
    )

    logger.info("Testing beat-synchronous segmentation (3.2)...")
    beat_feats, beat_frames = extractor.segment_audio_beat_sync(audio)
    logger.info(
        "Beat-synchronous features shape: %s, beat frames: %d",
        tuple(beat_feats.shape),
        len(beat_frames),
    )
    assert beat_feats.shape[1] == extractor.node_feature_dim, (
        f"Beat-sync width {beat_feats.shape[1]} != windowed width "
        f"{extractor.node_feature_dim}; gnn_model in_dim would break on mode switch"
    )

    logger.info("audio_features.py sanity check PASSED successfully.")