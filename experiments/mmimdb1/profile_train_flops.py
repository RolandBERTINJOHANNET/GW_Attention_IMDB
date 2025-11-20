#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_train_flops.py — Per-step (fwd+loss+bwd+optim) FLOPs, then total = per_step * MAX_STEPS
Profiles BLIP2 for selector ∈ {"shared","random"} using your exact GW instantiation.
"""

from __future__ import annotations
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from lightning.pytorch import Trainer

# === import your exact bits so shapes & hyperparams match ===
from function_train_attgw import (
    _make_dm_for, make_encoder, make_decoder,
    VIEW_MODE, INCLUDE_LABELS, BATCH_SIZE, MAX_STEPS, PRECISION,
    WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT,
    LR, WEIGHT_DECAY, SCHED_FACTORY,
)
BATCH_SIZE = 1
from shimmer import BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from shimmer.modules.selection import RandomSelection
from domains import BasicImageDomain, TextDomain, LabelDomain, CleanTargetDomain

GRAD_ACCUM = 1  # adjust if you use accumulate_grad_batches

def _human(n: float) -> str:
    if not n: return "0"
    units = [("F",1),("KF",1e3),("MF",1e6),("GF",1e9),("TF",1e12),("PF",1e15),("EF",1e18)]
    for name, scale in reversed(units):
        if n >= scale: return f"{n/scale:.3f} {name}"
    return f"{n:.3f} F"

def _sum_flops(prof: torch.profiler.profile) -> int:
    total = 0
    for evt in prof.key_averages():
        fl = getattr(evt, "flops", None)
        if fl is not None:
            total += int(fl)
    return total

def _build_dm(backbone: str):
    dm = _make_dm_for(
        backbone,
        batch_size=BATCH_SIZE, num_workers=9, pin_memory=True,
        normalize="none", view_mode=VIEW_MODE, include_labels=INCLUDE_LABELS,
    )
    dm.prepare_data(); dm.setup("fit")
    return dm

def _build_model(dm, backbone: str, selector: str, random_temperature: float = 0.1) -> GlobalWorkspaceFusion:
    pair_key = frozenset(["image_latents", "caption_embeddings"])
    lab_key  = frozenset(["labels"])
    train_pair = dm.train_datasets[pair_key]
    img_dim = int(train_pair.domain_data["image_latents"].shape[1])
    txt_dim = int(train_pair.domain_data["caption_embeddings"].shape[1])
    lbl_dim = int(dm.train_datasets[lab_key].domain_data["labels"].shape[1])

    domain_mods = {
        "image_latents":            BasicImageDomain(latent_dim=img_dim),
        "caption_embeddings":       TextDomain(latent_dim=txt_dim),
        "image_latents_clean":      CleanTargetDomain(latent_dim=img_dim, name="image_latents_clean"),
        "caption_embeddings_clean": CleanTargetDomain(latent_dim=txt_dim, name="caption_embeddings_clean"),
        "labels":                   LabelDomain(num_classes=lbl_dim),
    }
    gw_encoders = {
        "image_latents":      make_encoder(img_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "caption_embeddings": make_encoder(txt_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "labels":             make_encoder(lbl_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
    }
    gw_decoders = {
        "image_latents":      make_decoder(img_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "caption_embeddings": make_decoder(txt_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "labels":             make_decoder(lbl_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
    }
    loss_coefs = BroadcastLossCoefs(
        translations=1.0, demi_cycles=0.5, cycles=0.5, contrastives=0.05, fused=1.0
    )
    model = GlobalWorkspaceFusion(
        domain_mods=domain_mods,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=WORKSPACE_DIM,
        loss_coefs=loss_coefs,
        optim_lr=LR,
        optim_weight_decay=WEIGHT_DECAY,
        scheduler=SCHED_FACTORY(MAX_STEPS),
        scheduler_args=None,
        per_domain_keys=False,   # keys=shared baseline
    )
    if selector == "random":
        model.selection_mod = RandomSelection(temperature=float(random_temperature))
    return model

def _make_trainer():
    return Trainer(
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,  # no checkpoints; avoids misconfiguration
        log_every_n_steps=10_000,
        precision=PRECISION,
        inference_mode=False,
        accumulate_grad_batches=GRAD_ACCUM,
    )
# ... imports unchanged ...

def _build_dm_once(backbone: str):
    dm = _make_dm_for(
        backbone,
        batch_size=BATCH_SIZE, num_workers=0,  # <= fewer workers for profiling
        pin_memory=False,                      # <= reduce pinned CPU RAM
        normalize="none", view_mode=VIEW_MODE, include_labels=INCLUDE_LABELS,
    )
    dm.prepare_data(); dm.setup("fit")
    return dm

def profile_one_step(dm, backbone: str, selector: str, random_temperature: float = 0.1):
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Profiling | backbone={backbone} | selector={selector} | device={device} ===")
    print(f"[info] BATCH_SIZE={BATCH_SIZE} | MAX_STEPS={MAX_STEPS} | precision={PRECISION}")

    # Build model once; reuse the SAME dm (no re-loading huge arrays)
    print("[warmup] Building model…")
    model = _build_model(dm, backbone, selector, random_temperature)

    # Warm-up (unprofiled)
    print("[warmup] One step…")
    _make_trainer().fit(model=model, datamodule=dm)

    # Profiled step: reuse dm, rebuild ONLY the model
    print("[profile] Rebuilding model…")
    del model; gc.collect(); torch.cuda.empty_cache()
    model = _build_model(dm, backbone, selector, random_temperature)
    trainer = _make_trainer()

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    print("[profile] Starting profiler (one step)…")
    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        trainer.fit(model=model, datamodule=dm)

    per_step = float(_sum_flops(prof))
    total = per_step * MAX_STEPS
    print(f"\nPer-step FLOPs: {_human(per_step)}")
    print(f"Total training: {_human(total)}")

    # Cleanup to free RAM before next selector
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    return {"per_step_flops": per_step, "total_train_flops": total}

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    backbone = "blip2"
    dm = _build_dm_once(backbone)   # <= single load reused for both selectors
    results = {}
    for selector in ("shared", "random"):
        results[selector] = profile_one_step(dm, backbone, selector, random_temperature=0.1)

    print("======== SUMMARY (BLIP2) ========")
    for sel, r in results.items():
        print(f"{sel:>7} | per-step: {_human(r['per_step_flops']):>12} | total: {_human(r['total_train_flops'])}")
    print("=================================")

if __name__ == "__main__":
    main()
