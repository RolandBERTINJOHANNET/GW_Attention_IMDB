# MM-IMDb 1.0 Data Pipeline (paper-aligned)

This folder contains the exact preprocessing scripts used for the paper’s MM-IMDb 1.0 experiments (CLIP/BLIP-2 backbones, 70-view SimCLR/BERT augmentations, 23-class labels).

## Artifacts produced
- `mm_imdb1_globals_aug70_clip_image.npy` (+ `.stats.npz`)
- `mm_imdb1_globals_aug70_clip_text.npy` (+ `.stats.npz`)
- `mm_imdb1_globals_aug70_blip2_image.npy` (+ `.stats.npz`)
- `mm_imdb1_globals_aug70_blip2_text.npy` (+ `.stats.npz`)
- `mm_imdb1_globals_aug70_manifest.json`, `mm_imdb1_globals_aug70_aug_index.csv`
- `labels_all_23.npy`, `ids_all.txt`, `class_names_23.json`
- (Optional) `mm_imdb1_globals_aug70_labels.npy` via label augmentation (not required; training duplicates labels across views internally)

## Step-by-step
1) **Prepare raw MM-IMDb 1.0**
   - Unpack the dataset so you have `.../unzip/imdb/dataset/` with matched `<stem>.jpg/.jpeg` posters and `<stem>.json` synopses.
   - Keep the official `split.json` in `.../unzip/imdb/` (train/dev/test lists by stem).

2) **Build 70-view CLIP/BLIP-2 embeddings**
   ```bash
   cd data/mmimdb1
   python make_mm_imdb1_clip_blip2_embeddings.py \
     --data-dir /path/to/imdb1/unzipped_imdb/imdb/dataset \
     --out-dir  /path/to/embeddings_clip_blip2_aug70
   ```
   - Models: CLIP image `openai/clip-vit-base-patch32`; CLIP text `sentence-transformers/clip-ViT-B-32-multilingual-v1`; BLIP-2 `Salesforce/blip2-flan-t5-xl`.
   - Augmentations: SimCLR (images) + BERT-style masking (text) for views 1..69; view 0 is clean.
   - Sorting: stems are sorted intersection of image/json; row order = `orig_idx * 70 + view`.
   - Per-modality z-score across the dataset is applied and stored in `.stats.npz`.

3) **Build clean labels (23 classes)**
   ```bash
   cd data/mmimdb1
   python build_labels_23.py \
     --dataset-dir /path/to/imdb1/unzipped_imdb/imdb/dataset \
     --out-dir    /path/to/labels_23
   ```
   - Produces `labels_all_23.npy` (float32 multi-hot), `ids_all.txt` (stems), `class_names_23.json`.
   - Order matches the embedding stems (sorted intersection of image/json).

4) **(Optional) Augment labels to 70 views**
   - Not required for training: `dataset_1x_aug70.py` in `experiments/mmimdb1/` duplicates base labels across views.
   - If you need noisy/augmented labels aligned 1:1 with latents:
     ```bash
     cd data/mmimdb1
     python augment_labels.py
     ```
     This writes `mm_imdb1_globals_aug70_labels.npy` alongside the embeddings, with deterministic per-view Gaussian noise.

## How the training scripts consume the data
- In `experiments/mmimdb1/function_train_attgw.py` the datamodule factory (`make_datamodule_clip_aug70` / `make_datamodule_blip2_aug70` from `dataset_1x_aug70.py`) expects:
  - Embeddings dir containing the four `mm_imdb1_globals_aug70_{clip|blip2}_{image|text}.npy` files (+ stats and manifest).
  - Raw dataset dir with `split.json` to build train/val/test indices (original-level). View selection defaults to `clean` (view 0 only), but the 70-view arrays are required for consistency with the paper’s extraction.
  - Labels dir with `labels_all_23.npy` and `ids_all.txt`; labels are duplicated across views inside the datamodule when `include_labels=True`.
- No additional normalization is applied in the datamodule (`normalize="none"` by default) because extractor z-scoring has already been done.

## Quick reference (paths/env)
- Set `OTHER_BACKBONES_DIR_IMDB1_AUG70` to your embeddings dir if you keep the defaults.
- Set `IMDB1_LABELS_DIR_23` to your labels dir (for training scripts that need labels).
- Set `MMIMDB1_DIR` to your raw dataset root (to resolve `split.json`).
