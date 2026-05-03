#!/usr/bin/env python3
"""Preprocess all GuitarSet tracks and save as .npz feature caches.

Run once after downloading the dataset. Each track produces one .npz file
containing the full (X, y) pair for all 6 strings stacked together.
Subsequent runs skip already-cached tracks, so it is safe to re-run if
interrupted.

Usage
-----
    # Download dataset on first run and cache to data/cache/
    python scripts/build_cache.py --download

    # Use a custom mirdata data directory
    python scripts/build_cache.py --data-home /path/to/guitarset --download

    # Build frequency-label cache instead of MIDI
    python scripts/build_cache.py --label frequency

    # Load a cached dataset later in a notebook or script:
    #   from rice_ml.preprocessing import load_dataset
    #   X, y = load_dataset("data/cache/03_BN3-119-G_solo.npz")
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rice_ml.preprocessing.download import get_guitarset, track_ids
from rice_ml.preprocessing.dataset import build_multi_string_dataset_from_track, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GuitarSet .npz feature cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-home", default=None,
        help="mirdata cache directory (default: ~/.mirdata/guitarset/)",
    )
    parser.add_argument(
        "--out-dir", default="data/cache",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download audio_hex_cln + annotations (~1.6 GB). Full dataset is ~7.5 GB but we only need these two.",
    )
    parser.add_argument(
        "--label", default="midi", choices=["midi", "voiced", "frequency"],
        help="Label type to assign to frames",
    )
    parser.add_argument(
        "--frame-len", type=int, default=2048,
        help="Samples per frame",
    )
    parser.add_argument(
        "--hop-len", type=int, default=512,
        help="Samples between frames",
    )
    parser.add_argument(
        "--n-mfcc", type=int, default=13,
        help="Number of MFCC coefficients",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initialising GuitarSet via mirdata...")
    gs = get_guitarset(data_home=args.data_home, download=args.download)
    ids = track_ids(gs)
    total = len(ids)
    print(f"Found {total} tracks. Saving to {out_dir}/\n")

    skipped = errors = saved = 0

    for i, tid in enumerate(ids, 1):
        out_path = out_dir / f"{tid}.npz"
        if out_path.exists():
            print(f"  [{i:3d}/{total}] {tid} — skipped (already cached)")
            skipped += 1
            continue
        try:
            track = gs.track(tid)
            X, y = build_multi_string_dataset_from_track(
                track,
                label=args.label,
                frame_len=args.frame_len,
                hop_len=args.hop_len,
                n_mfcc=args.n_mfcc,
            )
            save_dataset(X, y, out_path)
            print(f"  [{i:3d}/{total}] {tid} — {X.shape[0]:,} frames → {out_path.name}")
            saved += 1
        except Exception as exc:
            print(f"  [{i:3d}/{total}] {tid} — ERROR: {exc}")
            errors += 1

    print(f"\nDone. {saved} saved, {skipped} skipped, {errors} errors.")


if __name__ == "__main__":
    main()
