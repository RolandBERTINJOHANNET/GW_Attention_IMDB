# IMDB Attention GW (paper-aligned repro)

This repo hosts the five paper experiments (MM-IMDB 1.0) on top of the public shimmer codebase (submodule) plus the full data preprocessing pipeline (CLIP/BLIP-2 embeddings, labels).

## Layout
- `third_party/shimmer`: shimmer public submodule.
- `experiments/mmimdb1/`: training/eval entrypoints for the four runs (CLIP/BLIP-2 × random/attention) and the FLOPs script.
- `data/mmimdb1/`: data preprocessing scripts (raw → embeddings → labels).
- `outputs/`: ignored directory for checkpoints/runs/logs.

## Quick start
1) Pull submodule and deps
```bash
git submodule update --init --recursive
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e third_party/shimmer
```

2) Build data (paper pipeline, MM-IMDB 1.0)
```bash
# Paths
RAW_IMDB1=/path/to/imdb1/unzipped_imdb/imdb        # has dataset/ + split.json
EMB_DIR=/path/to/embeddings_clip_blip2_aug70       # where embeddings will go
LBL_DIR=/path/to/labels_23                         # where labels will go

# 2a. Embeddings (CLIP/BLIP-2, 70 views, SimCLR/BERT aug)
cd data/mmimdb1
python make_mm_imdb1_clip_blip2_embeddings.py \
  --data-dir "$RAW_IMDB1/dataset" \
  --out-dir  "$EMB_DIR"

# 2b. Labels (23-class multi-hot, clean)
python build_labels_23.py \
  --dataset-dir "$RAW_IMDB1/dataset" \
  --out-dir    "$LBL_DIR"
# Optional: per-view noisy labels (not required for training)
# python augment_labels.py
```

3) Run experiments (no end-to-end finetuning; labels are a modality)
```bash
cd experiments/mmimdb1
python - <<'PY'
from function_train_attgw import instantiate_and_train
# selector: "shared" (attention) or "random" (uniform fusion)
# backbone: "clip" or "blip2"
instantiate_and_train(seed=42, backbone="clip",  selector="random")  # Exp1
instantiate_and_train(seed=42, backbone="clip",  selector="shared")  # Exp2
instantiate_and_train(seed=42, backbone="blip2", selector="random")  # Exp3
instantiate_and_train(seed=42, backbone="blip2", selector="shared")  # Exp4
PY
# FLOPs (BLIP-2): profile_train_flops.py
python profile_train_flops.py
```

### Environment variables used by scripts
- `OTHER_BACKBONES_DIR_IMDB1_AUG70`: defaults to `/home/rbertin/attention/imdb1/embeddings_clip_blip2_aug70` (set to your `EMB_DIR`).
- `IMDB1_LABELS_DIR_23` or `LABELS_DIR_23`: defaults to `/home/rbertin/attention/imdb1/labels_23` (set to your `LBL_DIR`).
- `MMIMDB1_DIR`: defaults to `/home/rbertin/attention/imdb1/unzipped_imdb/imdb` (raw dataset root for `split.json`).
- Model IDs can be overridden in the embedding script via envs (e.g., `BLIP2_MODEL_ID`, `CLIP_MODEL_ID_TXT`).

### Notes
- Keep large blobs (.npy/.npz/.ckpt/.pt) out of git; use `data/` and `outputs/` locally.
- Secrets/keys should be set via environment variables only.
