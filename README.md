# IMDB Attention GW (experiments repo skeleton)

This folder is a standalone git repo meant to host the five cleaned experiments from the paper. It pulls the public shimmer repo as a submodule and layers experiment scripts/configs on top.

## Layout (planned)
- `third_party/shimmer`: shimmer public submodule (pinned commit/tag TBD).
- `experiments/`: cleaned training/eval/analysis entrypoints (to be added).
- `data/`: optional lightweight helpers or metadata (no large blobs committed).
- `outputs/`: ignored directory for checkpoints/runs/logs.

## Getting started
1. Initialize submodules (after cloning this repo):
   ```bash
   git submodule update --init --recursive
   ```
2. Install Python deps (from the imdb env snapshot):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Add the experiments and configs (next steps in this project).

## Notes
- Keep large data (.npy, .hdf5, posters, ckpts, runs) out of git; use the ignored `outputs/`/`data/` paths instead.
- Secrets (e.g., TMDB_API_KEY) should only be provided via environment variables, never committed.
