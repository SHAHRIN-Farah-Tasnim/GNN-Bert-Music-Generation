"""
GTZAN Preprocessing: Audio -> Segment Similarity Graphs
Walks raw GTZAN audio, extracts features via AudioFeatureExtractor,
builds one PyG graph per track, stratified split, saves .pt caches.
"""
from __future__ import annotations
import argparse, json, logging, os, random, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from audio_features import AudioFeatureExtractor, load_config, set_seed
from graph_builder import build_segment_graph

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preprocess_gtzan")

GTZAN_GENRES = ["blues", "classical", "country", "disco", "hiphop",
                "jazz", "metal", "pop", "reggae", "rock"]


def discover_tracks(audio_root, limit_per_genre=None):
    tracks = []
    for genre_idx, genre in enumerate(GTZAN_GENRES):
        genre_dir = os.path.join(audio_root, genre)
        if not os.path.isdir(genre_dir):
            logger.warning("Genre directory missing, skipping: %s", genre_dir)
            continue
        files = sorted(f for f in os.listdir(genre_dir)
                       if f.lower().endswith((".wav", ".au", ".mp3")))
        if limit_per_genre is not None:
            files = files[:limit_per_genre]
        for fname in files:
            tracks.append((os.path.join(genre_dir, fname), genre_idx,
                           os.path.splitext(fname)[0]))
    return tracks


def stratified_split(tracks, train_frac=0.7, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    by_genre = {}
    for item in tracks:
        by_genre.setdefault(item[1], []).append(item)
    splits = {"train": [], "val": [], "test": []}
    for genre_idx, items in by_genre.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])
    for name in splits:
        rng.shuffle(splits[name])
    return splits


def one_hot(genre_idx, num_classes=10):
    y = np.zeros(num_classes, dtype=np.float32)
    y[genre_idx] = 1.0
    return y


def build_graphs_for_split(split_name, items, extractor, sim_threshold,
                           add_temporal, num_classes=10):
    graphs, failures = [], []
    total = len(items)
    for i, (path, genre_idx, track_id) in enumerate(items, start=1):
        try:
            y_audio, _ = extractor.load_audio(path)
            segment_features = extractor.extract_segment_features(y_audio)
            if segment_features.shape[0] < 2:
                raise ValueError(f"only {segment_features.shape[0]} segment(s)")
            graph = build_segment_graph(
                segment_features=segment_features,
                similarity_threshold=sim_threshold,
                add_temporal_edges=add_temporal,
                labels=one_hot(genre_idx, num_classes),
                track_id=track_id,
            )
            graph.genre_idx = torch.tensor([genre_idx], dtype=torch.long)
            graph.genre = GTZAN_GENRES[genre_idx]
            graphs.append(graph)
        except Exception as exc:
            logger.warning("FAILED %s (%s): %s", track_id,
                           GTZAN_GENRES[genre_idx], exc)
            failures.append({"track_id": track_id, "path": path, "error": str(exc)})
        if i % 25 == 0 or i == total:
            logger.info("[%s] processed %d/%d tracks (%d graphs, %d failures)",
                        split_name, i, total, len(graphs), len(failures))
    return graphs, failures


def summarize_graphs(graphs):
    if not graphs:
        return {"num_graphs": 0}
    node_counts = [int(g.x.shape[0]) for g in graphs]
    edge_counts = [int(g.edge_index.shape[1]) for g in graphs]
    genre_counts = {}
    for g in graphs:
        genre_counts[g.genre] = genre_counts.get(g.genre, 0) + 1
    return {
        "num_graphs": len(graphs),
        "feature_dim": int(graphs[0].x.shape[1]),
        "nodes_mean": round(float(np.mean(node_counts)), 2),
        "nodes_min": int(np.min(node_counts)),
        "nodes_max": int(np.max(node_counts)),
        "edges_mean": round(float(np.mean(edge_counts)), 2),
        "genre_distribution": dict(sorted(genre_counts.items())),
    }


def export_sample_graphs(graphs, out_dir, n=20):
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for g in graphs[:n]:
        stem = os.path.join(out_dir, getattr(g, "track_id", f"graph_{written:03d}"))
        torch.save(g, f"{stem}.pt")
        payload = {
            "track_id": getattr(g, "track_id", None),
            "genre": getattr(g, "genre", None),
            "num_nodes": int(g.x.shape[0]),
            "feature_dim": int(g.x.shape[1]),
            "edge_index": g.edge_index.tolist(),
            "edge_attr": g.edge_attr.squeeze(-1).tolist() if g.edge_attr is not None else [],
            "y": g.y.tolist() if g.y is not None else [],
            "source": "GTZAN genres_original, real audio",
        }
        with open(f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Preprocess GTZAN into PyG graphs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--limit-per-genre", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--save-samples", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    paths_cfg = config.get("paths", {})
    raw_dir = paths_cfg.get("raw", "data/raw")
    processed_dir = paths_cfg.get("processed", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    audio_root = args.audio_root or os.path.join(raw_dir, "gtzan", "genres_original")
    if not os.path.isdir(audio_root):
        raise FileNotFoundError(f"GTZAN audio not found at '{audio_root}'")

    graph_cfg = config.get("graph", {})
    sim_threshold = graph_cfg.get("similarity_threshold", 0.7)
    add_temporal = graph_cfg.get("add_temporal_edges", True)

    extractor = AudioFeatureExtractor.from_config(args.config)
    logger.info("Extractor: segment_seconds=%.1f hop_seconds=%.1f node_dim=%d",
                extractor.segment_seconds, extractor.hop_seconds,
                extractor.node_feature_dim)

    tracks = discover_tracks(audio_root, args.limit_per_genre)
    if not tracks:
        raise RuntimeError(f"No audio files under '{audio_root}'")
    logger.info("Discovered %d tracks across %d genres", len(tracks),
                len({t[1] for t in tracks}))

    splits = stratified_split(tracks, args.train_frac, args.val_frac, seed)
    logger.info("Split sizes -> train: %d, val: %d, test: %d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    report = {
        "audio_root": audio_root, "seed": seed,
        "num_tracks_discovered": len(tracks),
        "segment_seconds": extractor.segment_seconds,
        "hop_seconds": extractor.hop_seconds,
        "similarity_threshold": sim_threshold,
        "add_temporal_edges": add_temporal,
        "num_classes": len(GTZAN_GENRES),
        "label_format": "10-dim one-hot (single-label genre)",
        "splits": {},
    }

    all_failures = []
    for split_name in ("train", "val", "test"):
        logger.info("--- Building %s graphs ---", split_name)
        graphs, failures = build_graphs_for_split(
            split_name, splits[split_name], extractor, sim_threshold,
            add_temporal, num_classes=len(GTZAN_GENRES))
        all_failures.extend(failures)
        out_path = os.path.join(processed_dir, f"{split_name}_graphs.pt")
        torch.save(graphs, out_path)
        stats = summarize_graphs(graphs)
        report["splits"][split_name] = {**stats, "num_failures": len(failures),
                                         "cache": out_path}
        logger.info("Saved %d %s graphs -> %s", len(graphs), split_name, out_path)
        logger.info("  %s", json.dumps(stats))
        if split_name == "train" and args.save_samples > 0:
            sample_dir = os.path.join(processed_dir, "graphs")
            n_written = export_sample_graphs(graphs, sample_dir, args.save_samples)
            logger.info("Exported %d real-audio example graphs -> %s",
                        n_written, sample_dir)
            report["sample_graphs_written"] = n_written

    report["failures"] = all_failures
    report_path = os.path.join(processed_dir, "gtzan_preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote preprocessing report -> %s", report_path)
    if all_failures:
        logger.warning("%d track(s) failed and were skipped.", len(all_failures))
    logger.info("Done. Train with: python src/train.py --task gnn_tag")


if __name__ == "__main__":
    main()
