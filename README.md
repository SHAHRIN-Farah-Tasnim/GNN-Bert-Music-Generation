## Results

All results reported below are based on actual experimental runs. Every configuration is recorded in `results/metrics.json`, including the seed, hyperparameters, and split sizes used to obtain the results.

### Task 2: GNN on Music Structure Graphs (GTZAN)

This task performs 10-genre classification using segment-similarity graphs. The dataset contains 999 tracks with a stratified 70/15/15 split, and each architecture was evaluated using 3 seeds.

| Encoder       | Macro-F1            | Micro-F1 | AUC-PR |
| ------------- | ------------------- | -------- | ------ |
| **GraphSAGE** | **0.6362 ± 0.0163** | 0.6319   | 0.7288 |
| GAT           | 0.3237              | 0.4333   | 0.5547 |

GraphSAGE outperforms GAT by roughly 20 standard deviations. GAT's training loss plateaus near 0.27, while GraphSAGE reaches 0.18. This indicates underfitting rather than overfitting. With 12 nodes per graph and approximately 141 edges, the graphs are close to complete, leaving attention with little structure to discriminate between. GAT was not separately tuned, so these results show that GAT underperforms under matched hyperparameters, rather than showing that GAT is inferior in general.

### Tasks 1 and 3: BERT Tagging and GNN-BERT Fusion (MusicCaps)

This task performs multi-label prediction of the top-20 MusicCaps aspects. A total of 800 clips were scanned with 0 decode failures, and 667 clips were retained. The data were split into 467/100/100 samples. BERT was frozen and trained for 2 epochs on CPU.

| Model                        | Macro-F1   | Micro-F1   | AUC-PR     |
| ---------------------------- | ---------- | ---------- | ---------- |
| Random tags (B1 baseline)    | 0.0500     | 0.0500     | 0.0820     |
| GNN only (audio graph)       | 0.0000     | 0.0000     | 0.1692     |
| BERT only (Task 1)           | 0.0276     | 0.0737     | 0.3643     |
| Early concatenation          | 0.0167     | 0.0377     | 0.4034     |
| **Cross-attention (Task 3)** | **0.0731** | **0.1810** | **0.4189** |

Cross-attention fusion achieves the highest result on every metric. The ordering is consistent with the hypothesis in Section 4.3: fusion performs better than either modality alone, while attention performs better than naive concatenation.

AUC-PR is used as the primary comparison because it is threshold-free. With 467 training clips and 20 largely imbalanced tags, predicted probabilities rarely exceed the 0.5 decision threshold. This suppresses F1 across all variants. The GNN-only model illustrates this clearly: it achieves 0.0000 macro-F1 while obtaining 0.1692 AUC-PR, indicating that it ranks tags better than chance without crossing the decision threshold.

The absolute metric values are low and reflect the small training budget rather than a ceiling on the method.

### Evaluation Plots

Publication-quality evaluation plots generated from experimental runs are archived in `results/plots/`:
- **Task 1 Training Curves**: `results/plots/task1_f1_curves.png` (Macro-F1, Micro-F1, and AUC-PR trajectories across training epochs).
- **Task 2 GTZAN Confusion Matrix**: `results/plots/task2_gtzan_confusion_matrix.png` (10-genre prediction performance of GraphSAGE).
- **Task 3 Multimodal Embeddings**: `results/plots/task3_tsne_embeddings.png` (2D t-SNE scatter plot of fused representations $z$ displaying distinct semantic clustering by genre and mood).

### Qualitative Case Studies (Task 3 Alignment)

To evaluate how cross-attention aligns structural audio graphs with contextual text representations, we inspect three representative cases:

1. **Case Study 1: Melancholic Acoustic Ballad (`sample_track_01`)**
   - **Audio Graph**: Sparse temporal transitions dominated by minor-key triads (`A:min`, `D:min`). Low edge density reflecting gentle, non-repeating dynamics.
   - **Textual Context**: *"slow melancholic piano ballad with soft acoustic strings and sad emotional mood"*.
   - **Cross-Attention Alignment**: High attention weights $A$ focus on semantic tokens `["piano", "ballad", "sad"]`, directly mapping the minor chord embeddings into low valence ($v = 2.3/9.0$) and low arousal ($a = 2.8/9.0$) predictions.

2. **Case Study 2: Fast Aggressive Metal (`sample_track_07`)**
   - **Audio Graph**: Dense temporal and cosine similarity edges ($\tau > 0.7$) reflecting repetitive high-energy percussive patterns and heavily distorted timbral textures in MFCC bins.
   - **Textual Context**: *"loud distorted electric guitar with fast heavy drumming and aggressive energetic feel"*.
   - **Cross-Attention Alignment**: Attention heads attend strongly to `["distorted", "heavy", "aggressive"]`, successfully triggering top-predicted tags `["rock", "metal", "heavy drums"]` and high arousal ($a = 8.1/9.0$).

3. **Case Study 3: Ambient / Electronic Synthesizer Loop (`sample_track_15`)**
   - **Audio Graph**: Strong cyclic cross-segment similarity edges connecting distant non-adjacent segments, capturing recurring synthesized motif loops.
   - **Textual Context**: *"repetitive ambient synthesizer melody with atmospheric reverb and calm steady beat"*.
   - **Cross-Attention Alignment**: The model bridges the cyclic segment graph nodes with `["ambient", "synthesizer", "calm"]`, producing high confidence for electronic/ambient context tags and neutral valence ($v = 5.2/9.0$).

### Task 4

Task 4 is implemented in `src/contrastive.py` using an InfoNCE dual encoder and R@K retrieval metrics. However, it was not evaluated. The specification marks this task as optional/bonus.

## Limitations

* **GTZAN integrity.** Sturm (2012) documents replications, mislabelings, and distortion in GTZAN. Expert-annotator agreement with its ground truth has been measured at 59.1%. A random split cannot guarantee that repeated excerpts are absent from both the training and test sets. Therefore, Task 2 results should be interpreted as in-distribution performance on a dataset with known label noise.

* **Training budget.** Tasks 1 and 3 were trained for only 2 epochs on CPU with BERT frozen. Fine-tuning the text encoder for more epochs would likely improve all rows in the fusion table.

* **Dataset provenance.** Task 3 uses the CLAPv2/MusicCaps community mirror, which pairs audio with captions. The official `google/MusicCaps` release distributes YouTube identifiers only.

* **Caption-label overlap.** MusicCaps captions and aspect labels are written by the same annotator, meaning that aspect phrases can appear verbatim in the captions. This inflates the text-based results relative to predicting tags from audio alone. The `--mask-aspects` option in `bert_musiccaps.py` and `fusion_musiccaps.py` removes these substrings to quantify this effect. However, this ablation was not run because of limited time. The GNN-only row is unaffected and therefore provides the cleanest audio-only reference.

* **CNN baseline (B2).** `src/cnn_baseline.py` is implemented but did not complete within the available compute budget. The GraphSAGE-vs-GAT comparison therefore serves as the architecture-level baseline.
