"""
Extract GuitarSet features to CSV format.

Produces two files:
  data/guitarset_full.csv    — all 360 tracks, all 6 strings, all frames
  data/guitarset_subset.csv  — 10-track subset (2 per genre: comp + solo,
                                random player per track, seed=42)

Columns:
  track_id, string_idx, rms, zcr, centroid, bandwidth, rolloff,
  mfcc_1 ... mfcc_13, midi_label

Run from repo root:
    python scripts/extract_csv_dataset.py
"""

import csv
import sys
import time
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from rice_Ml.preprocessing.guitarset import load_jams, get_note_events
from rice_Ml.preprocessing.audio import load_wav, extract_string, frame_signal, frame_center_times
from rice_Ml.preprocessing.features import extract_all
from rice_Ml.preprocessing.dataset import label_frames_midi

AUDIO_DIR = REPO_ROOT / "data" / "guitarset" / "audio_hex_cln"
ANNOT_DIR = REPO_ROOT / "data" / "guitarset" / "annotation"
OUT_DIR   = REPO_ROOT / "data"

FEATURE_NAMES = [
    "rms", "zcr", "centroid", "bandwidth", "rolloff",
    *[f"mfcc_{i}" for i in range(1, 14)],
]
CSV_HEADER = ["track_id", "string_idx"] + FEATURE_NAMES + ["midi_label"]

PLAYERS = ["00", "01", "02", "03", "04", "05"]

# One style-tempo-key per genre, comp + solo, random player each
GENRE_PIECES = [
    ("BN",   "BN1-129-Eb"),
    ("Funk", "Funk1-114-Ab"),
    ("Jazz", "Jazz1-130-D"),
    ("Rock", "Rock1-130-A"),
    ("SS",   "SS1-100-C#"),
]


def pick_subset_tracks(seed: int = 42) -> list[str]:
    """Pick 10 track IDs: 2 per genre (comp + solo), random distinct players."""
    rng = random.Random(seed)
    tracks = []
    for genre, piece in GENRE_PIECES:
        players = rng.sample(PLAYERS, 2)   # two distinct players
        tracks.append(f"{players[0]}_{piece}_comp")
        tracks.append(f"{players[1]}_{piece}_solo")
    return tracks


def get_all_pairs() -> list[tuple[str, Path, Path]]:
    """Return (track_id, wav_path, jams_path) for every available track."""
    pairs = []
    for wav in sorted(AUDIO_DIR.glob("*_hex_cln.wav")):
        tid = wav.name.replace("_hex_cln.wav", "")
        jams = ANNOT_DIR / f"{tid}.jams"
        if jams.exists():
            pairs.append((tid, wav, jams))
    return pairs


def extract_track(tid: str, wav: Path, jams_path: Path) -> list[list]:
    """Return CSV rows for one track (all 6 strings, all frames)."""
    jams = load_jams(jams_path)
    audio, sr = load_wav(wav)
    rows = []
    for s in range(6):
        mono   = extract_string(audio, string_idx=s)
        frames = frame_signal(mono, frame_len=2048, hop_len=512)
        times  = frame_center_times(frames.shape[0], frame_len=2048, hop_len=512, sr=sr)
        X      = extract_all(frames, sr=sr)
        events = get_note_events(jams, string_idx=s)
        labels = label_frames_midi(times, events)
        for feat_row, label in zip(X, labels):
            rows.append([tid, s] + feat_row.tolist() + [int(label)])
    return rows


def write_csv(out_path: Path, pairs: list[tuple[str, Path, Path]]) -> None:
    n = len(pairs)
    t0 = time.time()
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        total_rows = 0
        for i, (tid, wav, jams_path) in enumerate(pairs):
            rows = extract_track(tid, wav, jams_path)
            writer.writerows(rows)
            total_rows += len(rows)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(
                f"  [{i+1:>3}/{n}]  {tid:<35}  "
                f"+{len(rows):>5} rows  "
                f"total={total_rows:>8,}  "
                f"elapsed={elapsed:>5.0f}s  eta={eta:>5.0f}s",
                flush=True,
            )
    size_mb = out_path.stat().st_size / 1024**2
    print(f"\nWrote {total_rows:,} rows → {out_path.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = get_all_pairs()
    print(f"Found {len(all_pairs)} tracks.\n")

    # --- Subset ---
    subset_ids = set(pick_subset_tracks(seed=42))
    print("Subset tracks selected (seed=42):")
    for tid in sorted(subset_ids):
        print(f"  {tid}")
    print()

    subset_pairs = [(tid, w, j) for tid, w, j in all_pairs if tid in subset_ids]
    missing = subset_ids - {tid for tid, *_ in subset_pairs}
    if missing:
        print(f"WARNING: subset tracks not found on disk: {missing}")

    subset_out = OUT_DIR / "guitarset_subset.csv"
    if subset_out.exists():
        print(f"Subset CSV already exists ({subset_out.stat().st_size/1024**2:.1f} MB), skipping.")
    else:
        print(f"Extracting subset ({len(subset_pairs)} tracks) → {subset_out.name}")
        write_csv(subset_out, subset_pairs)

    print()

    # --- Full ---
    full_out = OUT_DIR / "guitarset_full.csv"
    if full_out.exists():
        print(f"Full CSV already exists ({full_out.stat().st_size/1024**2:.1f} MB), skipping.")
    else:
        print(f"Extracting full dataset ({len(all_pairs)} tracks) → {full_out.name}")
        write_csv(full_out, all_pairs)

    print("\nDone.")
