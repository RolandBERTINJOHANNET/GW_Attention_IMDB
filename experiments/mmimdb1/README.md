# MM-IMDb 1.0 (aug-70) experiments

This folder contains the exact scripts used for the five reported runs:

1) CLIP backbone, random fusion (no learned attention)
2) CLIP backbone, learned attention (shared-key selector)
3) BLIP-2 backbone, random fusion
4) BLIP-2 backbone, learned attention (shared-key selector)
5) FLOPs/efficiency analysis (BLIP-2, random vs. attention)

## Files
- `function_train_attgw.py` — trains a GW with CLIP **or** BLIP-2 latents + labels.
  - Backbones: set `backbone` arg to `clip` or `blip2`.
  - Selector: `selector="shared"` (learned attention) or `selector="random"` (uniform softmax scores).
  - Architecture: ResMLP enc/dec stacks (H=384, 2 blocks, dropout 0.1) into a 512-dim GW; labels are a third modality; clean targets supervise augmented views.
  - Logging/checkpoints: CSV + optional W&B; checkpoints in `final_1x_gw_models_atttest/<backbone>/...`.
- `dataset_1x_aug70.py` — datamodule for MM-IMDb 1.0 with 70-view augmentation.
  - Loads CLIP/BLIP-2 latents (aug-70), replicates labels across views, builds clean target mirrors, and exposes CombinedLoader splits per `split.json`.
- `domains.py` — domain wrappers: pass-through image/text enc/dec, BCE-with-logits label domain, clean-target domains (supervision only).
- `profile_train_flops.py` — torch.profiler-based FLOPs estimator for BLIP-2 models using the same config; toggles selector ∈ {shared, random} and scales per-step FLOPs by `MAX_STEPS`.

## How to run
```bash
# install shimmer submodule into the environment
pip install -e ../third_party/shimmer

# Train (import the helper)
python - <<'PY'
from function_train_attgw import instantiate_and_train
# Options:
#   backbone: "clip" | "blip2"
#   selector: "shared" (learned attention) | "random" (uniform fusion)
instantiate_and_train(seed=42, backbone="clip",  selector="random")
instantiate_and_train(seed=42, backbone="clip",  selector="shared")
instantiate_and_train(seed=42, backbone="blip2", selector="random")
instantiate_and_train(seed=42, backbone="blip2", selector="shared")
PY

# FLOPs analysis (BLIP-2 only; selector toggled inside profile_train_flops.py)
python profile_train_flops.py
```

Environment defaults point to the existing aug-70 latents and labels under `/home/rbertin/attention/imdb1/…`; override with env vars in `dataset_1x_aug70.py` if your paths differ.
