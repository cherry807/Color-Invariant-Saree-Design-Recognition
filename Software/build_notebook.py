"""Generate the self-contained Kaggle and Colab notebooks from the sareeid/ package sources.

  python tools/build_notebook.py   ->  notebooks/saree_colorinvariant_{kaggle,colab}.ipynb

The package files are embedded with %%writefile, so the notebooks run as-is without cloning
anything, and can never drift from the repository code. Only the setup/data cells differ.
"""
import json
import subprocess
import sys
from pathlib import Path

if len(sys.argv) == 1:  # build both targets
    for t in ("kaggle", "colab"):
        subprocess.run([sys.executable, __file__, t], check=True)
    sys.exit()
TARGET = sys.argv[1]

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "recolor", "data", "model", "losses", "benchmark", "metrics", "evaluate",
           "train", "infer", "efficiency", "prepare"]

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s, hidden=False):
    meta = {"_kg_hide-input": True, "jupyter": {"source_hidden": True}} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


if TARGET == "kaggle":
    md("""
# AIE-CASE: color-invariant saree design recognition

The task is face recognition for textiles: find the saree whose **design** matches a query photo, regardless of the **colorway** it was woven in.

**Before running:** *Settings → Accelerator: GPU (T4 or P100)*, *Internet: On* (pretrained weights), and *Add Input*:
1. `div456/indian-saree-patterns` (Kaggle)
2. the DeepLure Drive corpus, uploaded as a **private** Kaggle dataset (it is proprietary; do not make it public)

Every image folder under `/kaggle/input` is found automatically. Then click **Run All**. The default config takes about 1–1.5 h on a T4.
""")
else:
    md("""
# AIE-CASE: color-invariant saree design recognition

The task is face recognition for textiles: find the saree whose **design** matches a query photo, regardless of the **colorway** it was woven in.

**Before running:**
1. *Runtime → Change runtime type → T4 GPU*.
2. Open the DeepLure Drive folder, click **▾ next to "sarees_dataset" → Organise → Add shortcut → My Drive**. The notebook reads it straight from your Drive; nothing is re-uploaded anywhere.
3. *(optional)* Add a Colab secret 🔑 `KAGGLE_API_TOKEN` (your Kaggle token) to also use `div456/indian-saree-patterns`. Public datasets usually download without it.

Then *Runtime → Run all*. Checkpoints and results are saved to `MyDrive/saree_runs`, so a disconnect loses nothing.
""")

md("""
## Approach note (≤500 characters)
ConvNeXt-Atto (ImageNet) → GeM → 256-d L2 embedding. There are no design labels, so each deduplicated image is its own design. Training uses SupCon on 4 views per image, each with a random crop, a synthetic recolor (k-means Lab palette remap/permute/transfer, HSV, channel, gray) and capture noise. A palette bank shared across the batch puts different motifs in identical colors, so color cannot separate classes. Inference: resize, center crop, flip TTA, cosine, per-design max.
""")

code("""
import os, sys, subprocess, torch
print(torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm'], check=True); import timm
print('timm', timm.__version__)
os.makedirs('sareeid', exist_ok=True)
NW = os.cpu_count()  # dataloader workers
""" + ("""
W = '/kaggle/working'; WORK = f'{W}/work'
""" if TARGET == "kaggle" else """
from google.colab import drive
drive.mount('/content/drive')
W = '/content/drive/MyDrive/saree_runs'   # runs + results persist on Drive
WORK = '/content/work'                    # image cache on fast local disk
os.makedirs(W, exist_ok=True)
"""))

md("## Package source\nThe same code as the `sareeid/` package in the repo, written to disk here so the notebook is self-contained. Cells are collapsed.")
for m in MODULES:
    src = (ROOT / "sareeid" / f"{m}.py").read_text(encoding="utf8")
    code(f"%%writefile sareeid/{m}.py\n{src}", hidden=True)

md("""
## 1. Data
Pipeline: every image folder under `/kaggle/input` → decode (corrupt and <96 px files dropped) → byte-level dedupe (MD5) → resize to short side 320 and cache → **near-duplicate grouping** with a perceptual hash matched across all 8 rotations/flips → **70/10/20 split by group**.

Grouping by near-duplicate means a re-shot, rotated or re-uploaded copy of a test saree can never appear in train.
""")
if TARGET == "kaggle":
    code("""
import glob
from pathlib import Path
IMG = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
ROOTS = sorted({str(Path(p).parents[0]) for p in glob.glob('/kaggle/input/**/*', recursive=True)
                if Path(p).suffix.lower() in IMG})
# collapse to top-level dataset dirs so 'source' is the dataset name
ROOTS = sorted({'/'.join(r.split('/')[:4]) for r in ROOTS})
print(*ROOTS, sep='\\n')
assert ROOTS, 'No images found: add the datasets via "Add Input"'
""")
else:
    code("""
import glob
from pathlib import Path
# DeepLure corpus: the 'sarees_dataset' shortcut in My Drive (searched one level deep as a fallback)
cands = glob.glob('/content/drive/MyDrive/sarees_dataset') + glob.glob('/content/drive/MyDrive/*/sarees_dataset')
DEEPLURE = cands[0] if cands else None
print('DeepLure corpus:', DEEPLURE or 'NOT FOUND - add the Drive shortcut (see top cell)')
# Kaggle saree patterns via kagglehub
try:
    from google.colab import userdata
    os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
except Exception:
    pass
try:
    import kagglehub
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'], check=True); import kagglehub
try:
    KAGGLE_DIR = kagglehub.dataset_download('div456/indian-saree-patterns')
except Exception as e:
    KAGGLE_DIR = None; print('Kaggle dataset skipped:', e)
print('Kaggle dataset:', KAGGLE_DIR)
ROOTS = [r for r in (DEEPLURE, KAGGLE_DIR) if r]
assert ROOTS, 'No data found'
""")
code("""
import pandas as pd, importlib, sareeid.data
importlib.reload(sareeid.data)  # pick up the freshly written package file
print('dataset roots:', ROOTS)
sareeid.data.prepare(ROOTS, WORK)
man = pd.read_csv(f'{WORK}/manifest.csv')
display(man.groupby(['source','split']).size().unstack(fill_value=0))
gs = man.groupby('group').size()
print(f'{len(man)} images, {len(gs)} design groups, {(gs>1).sum()} groups with >1 near-duplicate photo')
""")

md("## 2. Synthetic colorways\nEach row is one design. Columns: original, then each recolor family. The last two (**curves**, **reinhard**) are *held out*: they are never used in training, only in evaluation.")
code("""
import numpy as np, matplotlib.pyplot as plt
from sareeid import recolor as rc
from sareeid.data import read_rgb, center_crop
rng = np.random.default_rng(1)
paths = man[man.split=='test'].path.sample(4, random_state=0).tolist()
donor = center_crop(read_rgb(man.path.iloc[0]), 224); pal = rc.extract_palette(donor)
fams = ['original'] + list(rc.TRAIN_FAMILIES + rc.HELDOUT_FAMILIES)
fig, ax = plt.subplots(len(paths), len(fams), figsize=(2*len(fams), 2*len(paths)))
for r, p in enumerate(paths):
    im = center_crop(read_rgb(p), 224)
    for c, f in enumerate(fams):
        ax[r, c].imshow(im if f == 'original' else rc.apply_family(im, f, rng, target=pal, ref=donor))
        ax[r, c].axis('off'); ax[0, c].set_title(f, fontsize=9)
plt.tight_layout(); plt.show()
""")

md("""
## 3. Evaluation protocol
Everything below runs on the held-out **test** split (designs never seen in training). The verification threshold is calibrated on **val** and applied unchanged to test.

| Protocol | Gallery | Query | What it measures |
|---|---|---|---|
| **P1 colorway** | 1 clean image per design | every design × 8 recolor families + crop/rotate/blur/JPEG | identification under recolor; separate score for the 2 held-out families |
| **P2 color trap** | same | design *d* painted in the exact palette of another design *d'* | R@1, and **Trap@1** = how often the palette donor is ranked first (color bias) |
| **P3 shared palette** | all designs in palette A | all designs in palette B | color carries zero information; only the motif can identify |
| **P4 re-photo** | same | real 2nd photos from near-duplicate groups | real capture variation (only when such groups exist) |
| **Verification** | pairs | genuine (cross-palette) vs impostor (random) and impostor (**same palette, different motif**) | AUC, EER, TAR@FAR=1%, accuracy at val threshold |

Every benchmark image comes from a seeded spec, so all models see byte-identical benchmarks.
""")

md("## 4. Baselines (no training)\nThe ImageNet backbone used as-is, on RGB input and on grayscale input. Grayscale is the naive fix for color invariance; it fails when two different dyes have the same luminance, and when colorways invert light/dark.")
code("""
BACKBONE = 'convnext_atto.d2_in1k'
!python -m sareeid.evaluate --manifest {WORK}/manifest.csv --zero-shot {BACKBONE} --out {W}/results/zs_rgb.json --workers {NW}
!python -m sareeid.evaluate --manifest {WORK}/manifest.csv --zero-shot {BACKBONE} --gray --out {W}/results/zs_gray.json --workers {NW}
""")

md("## 5. Train\nPK batches of 48 designs × 4 views; SupCon with τ=0.07; AdamW (head 3e-4, backbone 0.3×), cosine schedule with 1 warmup epoch, AMP. The checkpoint is selected on the val benchmark (mean R@1 of P1, P2 and P3).")
code("""
EPOCHS = 30
!python -m sareeid.train --manifest {WORK}/manifest.csv --out {W}/runs/atto_rgb --backbone {BACKBONE} --epochs {EPOCHS} --P 48 --views 4 --eval-every 5 --workers {NW}
""")
code("""
import json
h = json.load(open(f'{W}/runs/atto_rgb/history.json'))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot([r['epoch'] for r in h], [r['loss'] for r in h]); ax[0].set_title('train SupCon loss'); ax[0].set_xlabel('epoch')
ev = [r for r in h if 'val' in r]
for k in ['P1 R@1', 'P2 R@1', 'P3 R@1', 'Same-palette AUC']:
    ax[1].plot([r['epoch'] for r in ev], [r['val'][k] for r in ev], marker='o', label=k)
ax[1].set_title('val benchmark'); ax[1].legend(); ax[1].set_xlabel('epoch'); plt.show()
""")
code("""
!python -m sareeid.evaluate --manifest {WORK}/manifest.csv --ckpt {W}/runs/atto_rgb/best.pt --out {W}/results/atto_rgb.json --workers {NW}
""")

md("### Ablations (optional, set `RUN_ABLATIONS=True`)\n- **gray**: the same training on grayscale input. Does the RGB model actually use chromatic edges?\n- **no palette bank**: palette-transfer targets drawn independently per sample instead of from a shared bank (the `--bank 0` path).\n- **bigger backbone**: ConvNeXt-Tiny (28M). How much accuracy does the lean model give up?")
code("""
RUN_ABLATIONS = False
if RUN_ABLATIONS:
    !python -m sareeid.train --manifest {WORK}/manifest.csv --out {W}/runs/atto_gray --backbone {BACKBONE} --epochs {EPOCHS} --gray --eval-every 5 --workers {NW}
    !python -m sareeid.evaluate --manifest {WORK}/manifest.csv --ckpt {W}/runs/atto_gray/best.pt --out {W}/results/atto_gray.json --workers {NW}
    !python -m sareeid.train --manifest {WORK}/manifest.csv --out {W}/runs/atto_nobank --backbone {BACKBONE} --epochs {EPOCHS} --bank 0 --eval-every 5 --workers {NW}
    !python -m sareeid.evaluate --manifest {WORK}/manifest.csv --ckpt {W}/runs/atto_nobank/best.pt --out {W}/results/atto_nobank.json --workers {NW}
    !python -m sareeid.train --manifest {WORK}/manifest.csv --out {W}/runs/tiny_rgb --backbone convnext_tiny.in12k_ft_in1k --epochs {EPOCHS} --P 32 --eval-every 5 --workers {NW}
    !python -m sareeid.evaluate --manifest {WORK}/manifest.csv --ckpt {W}/runs/tiny_rgb/best.pt --out {W}/results/tiny_rgb.json --workers {NW}
""")

md("## 6. Results (test split)")
code("""
rows = {}
for f in sorted(glob.glob(f'{W}/results/*.json')):
    r = json.load(open(f)); rows[Path(f).stem] = r['headline']
res = pd.DataFrame(rows).T
display(res.style.format('{:.3f}').background_gradient(axis=0, cmap='Greens'))
print(res.to_markdown(floatfmt='.3f'))
""")
code("""
full = json.load(open(f'{W}/results/atto_rgb.json'))['full']
zs = json.load(open(f'{W}/results/zs_rgb.json'))['full']
fam = list(full['P1_per_family'])
x = np.arange(len(fam))
plt.figure(figsize=(10, 3.5))
plt.bar(x-0.2, [zs['P1_per_family'][f]['R@1'] for f in fam], 0.4, label='zero-shot RGB')
plt.bar(x+0.2, [full['P1_per_family'][f]['R@1'] for f in fam], 0.4, label='ours')
plt.xticks(x, [f + (' *' if f in rc.HELDOUT_FAMILIES else '') for f in fam], rotation=20)
plt.ylabel('R@1'); plt.title('P1 R@1 per recolor family (* = held out from training)'); plt.legend(); plt.show()
for name, r in [('zero-shot', zs), ('ours', full)]:
    print(name, 'cosine mean±sd', {k: f'{m:.2f}±{s:.2f}' for k, (m, s) in r['score_stats'].items()})
""")

md("## 7. Qualitative check: color-trap queries\nEach query is a test design painted in another design's palette. Green = the correct design, red = the palette donor (a color-biased model's answer).")
code("""
import torch
from sareeid.benchmark import build_specs, SpecDataset, embed_specs
from sareeid.data import load_manifest
from sareeid.model import load_checkpoint
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_items = load_manifest(f'{WORK}/manifest.csv', 'test')
B = build_specs(test_items, seed=0); S = B['specs']; ds = SpecDataset(S)
model, ck = load_checkpoint(f'{W}/runs/atto_rgb/best.pt'); model.to(dev)
idx = B['gallery'] + B['trap']
Z = embed_specs(model, [S[i] for i in idx], dev, workers=NW)
G, Q = Z[:len(B['gallery'])], Z[len(B['gallery']):]
sel = np.random.default_rng(0).choice(len(Q), 6, replace=False)
fig, ax = plt.subplots(len(sel), 6, figsize=(12, 2.1*len(sel)))
for r, qi in enumerate(sel):
    s = S[B['trap'][qi]]; top = (G @ Q[qi]).argsort(descending=True)[:5].tolist()
    ax[r, 0].imshow(ds.render(s)); ax[r, 0].set_title('query', fontsize=8)
    for c, g in enumerate(top, 1):
        ax[r, c].imshow(ds.render(S[B['gallery'][g]]))
        col = 'green' if g == s['label'] else 'red' if g == s['donor_label'] else 'gray'
        for sp in ax[r, c].spines.values(): sp.set_color(col); sp.set_linewidth(4)
        ax[r, c].set_title(f'#{c} {float(G[g] @ Q[qi]):.2f}', fontsize=8, color=col)
    for a in ax[r]: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
""")

md("## 8. Efficiency")
code("""
effs = {}
for run in sorted(glob.glob(f'{W}/runs/*/best.pt')):
    out = f'{W}/results/eff_{Path(run).parent.name}.json'
    !python -m sareeid.efficiency --ckpt {run} --out {out}
    effs[Path(run).parent.name] = json.load(open(out))
display(pd.DataFrame(effs).T)
""")

md("## 9. Inference API\nIndex a gallery folder (sub-folder name = design id, so several colorways of one design can sit in one folder), then identify and verify.")
code("""
from sareeid.infer import SareeMatcher
import shutil, cv2
m = SareeMatcher(f'{W}/runs/atto_rgb/best.pt')
gal = Path(f'{W}/demo_gallery'); shutil.rmtree(gal, ignore_errors=True); gal.mkdir()
for it in test_items[:200]:
    (gal / f'design_{it.group}').mkdir(exist_ok=True); shutil.copy(it.path, gal / f'design_{it.group}')
index = m.build_index(str(gal))
q = rc.palette_remap(read_rgb(test_items[7].path), np.random.default_rng(3), mode='random')
cv2.imwrite(f'{W}/query.jpg', cv2.cvtColor(q, cv2.COLOR_RGB2BGR))
print('truth: design', test_items[7].group)
for lab, s, p in m.identify(f'{W}/query.jpg', index, topk=5): print(f'{lab:>14s}  {s:.3f}')
print('verify(query, its original):', m.verify(f'{W}/query.jpg', test_items[7].path))
print('verify(query, other design):', m.verify(f'{W}/query.jpg', test_items[8].path))
""")

md("## 10. Clean up\nThe DeepLure corpus is proprietary. Once the exercise is done, delete the private Kaggle dataset and `/kaggle/working/work/cache`.")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
if TARGET == "colab":
    nb["metadata"]["accelerator"] = "GPU"
    nb["metadata"]["colab"] = {"gpuType": "T4", "provenance": []}
for i, c in enumerate(nb["cells"]):  # nbformat 4.5: list of lines + cell id
    c["id"] = f"c{i:03d}"
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
out = ROOT / "notebooks" / f"saree_colorinvariant_{TARGET}.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf8")
print("wrote", out)
