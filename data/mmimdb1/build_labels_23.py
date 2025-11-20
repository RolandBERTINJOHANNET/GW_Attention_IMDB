#!/usr/bin/env python3
"""
Build 23-class multi-hot labels for MM-IMDb 1.0 (clean, no aug).

- Reads all JSON files in the raw dataset folder (aligned with posters).
- Uses the canonical 23 genre list from the paper (ordered as in prior runs).
- Writes:
    labels_all_23.npy  (float32 multi-hot, shape [N, 23], N = #aligned samples)
    ids_all.txt        (sorted stems matching row order)
    class_names_23.json

Assumptions:
- Raw dataset dir contains matching <stem>.json and <stem>.jpg/.jpeg files.
- Stems are aligned lexicographically to match the embedding extraction order.

Usage:
    python build_labels_23.py \
      --dataset-dir /path/to/imdb1/unzipped_imdb/imdb/dataset \
      --out-dir /path/to/labels_23
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

CANON_23 = [
    "Action","Adventure","Animation","Biography","Comedy","Crime","Documentary",
    "Drama","Family","Fantasy","Film-Noir","History","Horror","Music","Musical",
    "Mystery","Romance","Sci-Fi","Short","Sport","Thriller","War","Western",
]
CANON_MAP = {g.lower(): i for i, g in enumerate(CANON_23)}

def list_by_suffix(root: Path, exts: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for e in exts:
        out += list(root.glob(f"*{e}"))
        out += list(root.glob(f"*{e.upper()}"))
    out.sort()
    return out

def parse_genres(obj: dict) -> list[str]:
    for key in ("genres", "Genres", "genre", "labels"):
        val = obj.get(key)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Path to imdb1 dataset folder with .json/.jpg")
    ap.add_argument("--out-dir",      required=True, help="Where to write labels_all_23.npy, ids_all.txt, class_names_23.json")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = list_by_suffix(ds, (".jpg", ".jpeg"))
    jsons  = list_by_suffix(ds, (".json",))
    img_map = {p.stem: p for p in images}
    js_map  = {p.stem: p for p in jsons}
    stems = sorted(set(img_map) & set(js_map))
    if not stems:
        raise FileNotFoundError(f"No aligned image/json stems in {ds}")
    if len(stems) < max(len(images), len(jsons)):
        print(f"warning: dropping {max(len(images), len(jsons)) - len(stems)} unmatched files")

    labels = []
    for s in stems:
        obj = json.loads(js_map[s].read_text(encoding="utf-8"))
        genres = parse_genres(obj)
        y = np.zeros(len(CANON_23), dtype=np.float32)
        for g in genres:
            gk = g.lower().replace("science fiction", "sci-fi").replace("sci fi", "sci-fi")
            idx = CANON_MAP.get(gk)
            if idx is not None:
                y[idx] = 1.0
        labels.append(y)

    Y = np.stack(labels, axis=0) if labels else np.zeros((0, len(CANON_23)), dtype=np.float32)
    np.save(out / "labels_all_23.npy", Y)
    (out / "ids_all.txt").write_text("\n".join(stems), encoding="utf-8")
    (out / "class_names_23.json").write_text(json.dumps(CANON_23, indent=2), encoding="utf-8")
    print(f"✓ wrote labels_all_23.npy shape={Y.shape} | ids_all.txt lines={len(stems)}")

if __name__ == "__main__":
    main()
