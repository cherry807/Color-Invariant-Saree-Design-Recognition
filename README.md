# Color-invariant saree design recognition (AIE-CASE)

Given a photo of a saree, find the matching **design** in a gallery (identification) or decide whether two photos show the same design (verification), **regardless of the colorway**. It works like face recognition for textiles.

**Run it:** both notebooks are self-contained and generated from the same package code (`tools/build_notebook.py`).
- **Colab:** [`notebooks/saree_colorinvariant_colab.ipynb`](notebooks/saree_colorinvariant_colab.ipynb). Use a T4 runtime. The DeepLure corpus is read from a Drive shortcut, the Kaggle set comes via `kagglehub`, and outputs go to `MyDrive/saree_runs`.
- **Kaggle:** [`notebooks/saree_colorinvariant_kaggle.ipynb`](notebooks/saree_colorinvariant_kaggle.ipynb). Turn on GPU and Internet and add both datasets as inputs.

Then Run All (about 1–1.5 h on a T4).

## Approach note (≤500 chars)

> ConvNeXt-Atto (ImageNet) → GeM → 256-d L2 embedding. There are no design labels, so each deduplicated image is its own design. Training uses SupCon on 4 views per image, each with a random crop, a synthetic recolor (k-means Lab palette remap/permute/transfer, HSV, channel, gray) and capture noise. A palette bank shared across the batch puts different motifs in identical colors, so color cannot separate classes. Inference: resize, center crop, flip TTA, cosine, per-design max.

## 1. Problem formulation

Neither dataset has *design* or *colorway* labels. The DeepLure corpus is a flat folder of close-up handloom shots, and the Kaggle set is organised by style, not by design. So the task is framed as **instance-level metric learning with synthetic colorways**:

- **identity** = one physical design (an image, or a group of near-duplicate photos of it)
- **positives** = the same design under a different colorway, crop, orientation or capture conditions
- **negatives** = any other design, *including other designs rendered in exactly the same palette*

The core requirement ("same motif in a different palette must match; different motifs in an identical palette must not") is then built directly into the training batches and into the benchmark.

### Why synthetic colorways are a faithful proxy
A woven colorway keeps the motif layout fixed and swaps which yarn color fills each region. [`recolor.py`](sareeid/recolor.py) simulates exactly that:
1. k-means (k=3–7) in CIELAB, with lightness down-weighted so a fold or shadow does not become a separate "dye".
2. Merge clusters closer than ΔE≈22, so one dye's shading is never split into two colors.
3. Repaint each region with a new color using a **soft** assignment, so the boundaries between dyes stay smooth.
4. Add back the per-pixel residual (weave texture, sheen, folds).

New colors come from one of: a random palette (colors kept ≥35 ΔE apart, so two motif regions never merge), a **permutation** of the image's own colors (gold-on-red ↔ red-on-gold), or a **transfer** of another saree's palette by lightness rank. Simpler shifts (HSV, channel shuffle/invert, gray) add variety.

Two families, **per-channel tone curves** and **Reinhard color transfer**, are *never used in training*. They exist only to test whether the learned invariance generalises beyond the training augmentations.

## 2. Pipeline

| Stage | What | Why |
|---|---|---|
| Ingest | recursive scan; drop unreadable files, images <96 px and MD5 byte-duplicates | the datasets are scraped and noisy |
| Cache | resize to short side 320 and store as JPEG | 10–50× faster data loading |
| Dedupe-group | 64-bit DCT pHash; pairs whose Hamming distance ≤6 under **any of the 8 rotations/flips** are union-found into one group. Roboflow copies (`<photo>_jpg.rf.<hash>.jpg`) of one source photo are also merged | re-uploads, rotated copies and Roboflow augmentations would otherwise leak across splits (training is rotation-invariant) |
| Re-split | the Kaggle set's own train/valid/test folders and style classes are ignored | they are not design-disjoint; we re-split by group |
| Split | 70/10/20 train/val/test **by group**, seed 0 | test designs are never seen in training |
| Train view | random resized crop (0.35–1) → recolor → flip / rot90 / small rotation / blur / uneven light / noise / JPEG → ImageNet norm | fabric has no canonical orientation; phone photos vary |
| Model | `convnext_atto.d2_in1k` → GeM (learnable p) → Linear(320→256) + BN → L2 | see §4 |
| Inference | resize short side → center crop 224 → embed (+ h-flip TTA) → cosine similarity; a gallery design scores the **max over its colorways** | more colorways in the gallery can only help |
| Verification | cosine ≥ τ, where τ = the EER threshold **calibrated on val** and stored in the checkpoint | the threshold is never tuned on test |

## 3. Training

- **Loss:** multi-positive SupCon (τ=0.07). Every view of the same design is a positive; every other view in the batch is a negative.
- **Sampling:** PK batches, P=48 designs × K=4 views (192 embeddings per step). Near-duplicate photos in the same group share a label, so they are extra *real* positives.
- **Shared palette bank:** each epoch draws 24 palettes (half from real sarees, half random), and every recolor target in the batch comes from this bank. Many different designs in a batch therefore wear the *same* palette and are negatives to each other, so color cannot separate classes in the loss. `--bank 0` is the ablation without it.
- **Optimiser:** AdamW, lr 3e-4 for the head and 0.3× that for the backbone, weight decay 0.05, 1 epoch of warmup then cosine decay, AMP, gradient clipping at 5, 30 epochs.
- **Model selection:** best mean R@1 of P1, P2 and P3 on the **val** benchmark.

## 4. Evaluation protocol

All protocols use the held-out **test** split. Each benchmark image is a seeded spec rendered on the fly, so every model sees a byte-identical benchmark ([`benchmark.py`](sareeid/benchmark.py)). The gallery is one clean, center-cropped image per test design.

| Protocol | Query | Metric | Tests |
|---|---|---|---|
| **P1 colorway** | every design × 8 recolor families, plus crop / rotation / blur / JPEG | R@1, R@5, R@10, mAP, overall and per family | color invariance; generalisation to the **held-out** families |
| **P2 color trap** | design *d* painted in the exact palette of another test design *d′* | R@1 and **Trap@1** (top-1 = palette donor) | how often the model is fooled by color |
| **P3 shared palette** | gallery: every design in palette A; queries: every design in palette B (3 draws) | R@1, mAP | color carries zero information |
| **P4 re-photo** | real second photos from near-duplicate groups | R@1, mAP | real capture variation (when such groups exist) |
| **Verification** | 2 genuine + 2 random-impostor + 2 **same-palette-impostor** pairs per design | ROC-AUC, EER, TAR@FAR=1%/0.1%, accuracy at the val threshold; also a *same-palette-only* AUC | the "different motif, identical palette must not match" requirement |

Baselines on the identical benchmark: the zero-shot ImageNet backbone on RGB input, and the same on **grayscale** input (the naive way to get color invariance). Optional ablations in the notebook: training on gray input, no palette bank, and ConvNeXt-Tiny (28M).

**Known limitation, stated up front:** the colorway queries are synthetic. P4 is the real-photo check, but it only covers re-shots, not real alternate colorways. The next step for a production system is ~100 hand-verified real colorway pairs; `SareeMatcher.verify` and `metrics.verification` already take arbitrary pairs.

## 5. Results (test split)

Paste the table printed by notebook §6 here.

| model | P1 R@1 | P1 R@1 held-out | P2 R@1 | P2 Trap@1 ↓ | P3 R@1 | Verif AUC | EER ↓ | TAR@FAR=1% | Same-palette AUC |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot RGB | | | | | | | | | |
| zero-shot gray | | | | | | | | | |
| **ours (Atto, 30 ep)** | | | | | | | | | |

## 6. Efficiency

| | ConvNeXt-Atto embedder (ours) |
|---|---|
| Parameters | **3.46 M** |
| Compute | **0.55 GMACs** (1.09 GFLOPs) at 224×224 |
| Embedding | **256-d**, stored as fp16 = **512 bytes** per image |
| CPU latency, batch 1 | 27.7 ms (10-thread desktop CPU, fp32, measured locally) |
| GPU latency / throughput | reported by notebook §8 (fp16) |

Why a lean model is enough here: motifs are local, high-contrast, repetitive structures, which is exactly what early and mid-level conv features capture. ImageNet pretraining already supplies them, and fine-tuning only has to *remove* the color dependence, not learn new features. GeM pooling emphasises sparse motif activations over the plain background. A 512-byte fp16 code means 1M gallery designs fit in 0.5 GB and can be searched with a single matrix multiply (or FAISS).

## 7. Repository

```
sareeid/
  recolor.py     colorway synthesis (train and held-out families)
  data.py        ingest, cache, pHash grouping, split, multi-view training set
  model.py       backbone + GeM + projection
  losses.py      multi-positive SupCon
  benchmark.py   deterministic P1–P4 and verification spec builder
  metrics.py     CMC / mAP / Trap@1, ROC-AUC / EER / TAR@FAR
  evaluate.py    val-calibrated evaluation CLI
  train.py       training CLI
  infer.py       index / identify / verify CLI and SareeMatcher API
  efficiency.py  params, FLOPs, latency
tools/build_notebook.py   regenerates the Kaggle notebook from the package
```

Local use:
```bash
pip install -r requirements.txt
python -m sareeid.prepare  --roots <dataset dirs...> --out work
python -m sareeid.train    --manifest work/manifest.csv --out runs/atto_rgb
python -m sareeid.evaluate --manifest work/manifest.csv --ckpt runs/atto_rgb/best.pt --out results/atto_rgb.json
python -m sareeid.infer index    --ckpt runs/atto_rgb/best.pt --images gallery/ --out gallery.pt
python -m sareeid.infer identify --ckpt runs/atto_rgb/best.pt --gallery gallery.pt --image q.jpg
```

## 8. Disclosures

- **Pretrained weights:** `timm/convnext_atto.d2_in1k` (ImageNet-1k, from Hugging Face via timm). The optional ablation uses `timm/convnext_tiny.in12k_ft_in1k`.
- **Data:** the DeepLure saree corpus (proprietary; used only in a private Kaggle dataset, not redistributed, to be deleted after the exercise) and Kaggle `div456/indian-saree-patterns`. No other external data.
- **Libraries:** PyTorch, timm, OpenCV, NumPy, scikit-learn (ROC), pandas and matplotlib (reporting).
