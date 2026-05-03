"""
Extract GuitarSet features for the Perceptron and MLP example notebooks.

Run from the repo root:
    python scripts/extract_example_features.py

What it does:
  1. Unzips data/audio_hex-pickup_debleeded*.zip  →  data/guitarset/audio_hex_cln/
     Unzips data/annotation.zip                   →  data/guitarset/annotation/
     (skips if already extracted)
  2. Builds voiced/unvoiced features for the Perceptron example (5 tracks, string 0)
     → examples/Supervised Learning/Perceptron/guitarset_features.npz
  3. Builds MIDI-note features for the MLP example (10 tracks, all 6 strings, top-12 notes)
     → examples/Supervised Learning/MLP/guitarset_features.npz
"""

import sys
import glob
import zipfile
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GUITARSET_DIR = DATA_DIR / "guitarset"
AUDIO_DIR = GUITARSET_DIR / "audio_hex_cln"
ANNOT_DIR = GUITARSET_DIR / "annotation"

PERCEPTRON_OUT = REPO_ROOT / "examples" / "Supervised Learning" / "Perceptron" / "guitarset_features.npz"
MLP_OUT = REPO_ROOT / "examples" / "Supervised Learning" / "MLP" / "guitarset_features.npz"

N_PERCEPTRON_TRACKS = 5
N_MLP_TRACKS = 10
TOP_N_NOTES = 12


def unzip_data(n_tracks: int = N_MLP_TRACKS) -> None:
    """Unzip only the first n_tracks files from each zip (to save disk space)."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    debleeded = sorted(DATA_DIR.glob("audio_hex-pickup_debleeded*.zip"))
    if not debleeded:
        print("ERROR: could not find audio_hex-pickup_debleeded*.zip in data/", file=sys.stderr)
        sys.exit(1)
    audio_zip = debleeded[0]
    annot_zip = DATA_DIR / "annotation.zip"
    if not annot_zip.exists():
        print("ERROR: data/annotation.zip not found.", file=sys.stderr)
        sys.exit(1)

    # Collect the track IDs for the first n_tracks from the audio zip
    with zipfile.ZipFile(audio_zip) as zf:
        wav_members = sorted(m for m in zf.namelist() if m.endswith("_hex_cln.wav"))
    track_ids = [Path(m).name.replace("_hex_cln.wav", "") for m in wav_members[:n_tracks]]

    # Extract only the needed audio files
    already_wav = len(list(AUDIO_DIR.glob("*.wav")))
    if already_wav < n_tracks:
        needed_wav = {f"{tid}_hex_cln.wav" for tid in track_ids}
        to_extract = [m for m in wav_members[:n_tracks] if Path(m).name not in
                      {f.name for f in AUDIO_DIR.glob("*.wav")}]
        if to_extract:
            print(f"Extracting {len(to_extract)} WAV files from {audio_zip.name} ...")
            with zipfile.ZipFile(audio_zip) as zf:
                for member in to_extract:
                    # Extract flat into AUDIO_DIR (strip any subdirectory)
                    data = zf.read(member)
                    out_path = AUDIO_DIR / Path(member).name
                    out_path.write_bytes(data)
            print(f"  {len(list(AUDIO_DIR.glob('*.wav')))} WAV files available.")
    else:
        print(f"Audio already extracted ({already_wav} files).")

    # Extract only the needed annotation files
    already_jams = len(list(ANNOT_DIR.glob("*.jams")))
    if already_jams < n_tracks:
        with zipfile.ZipFile(annot_zip) as zf:
            jams_members = {Path(m).name: m for m in zf.namelist() if m.endswith(".jams")}
        to_extract = [jams_members[f"{tid}.jams"] for tid in track_ids
                      if f"{tid}.jams" in jams_members and
                      not (ANNOT_DIR / f"{tid}.jams").exists()]
        if to_extract:
            print(f"Extracting {len(to_extract)} JAMS files from annotation.zip ...")
            with zipfile.ZipFile(annot_zip) as zf:
                for member in to_extract:
                    data = zf.read(member)
                    out_path = ANNOT_DIR / Path(member).name
                    out_path.write_bytes(data)
            print(f"  {len(list(ANNOT_DIR.glob('*.jams')))} JAMS files available.")
    else:
        print(f"Annotations already extracted ({already_jams} files).")


def get_track_pairs() -> list[tuple[Path, Path]]:
    """Return sorted list of (wav_path, jams_path) pairs."""
    pairs = []
    for wav in sorted(AUDIO_DIR.glob("*_hex_cln.wav")):
        track_id = wav.name.replace("_hex_cln.wav", "")
        jams = ANNOT_DIR / f"{track_id}.jams"
        if jams.exists():
            pairs.append((wav, jams))
    return pairs


def extract_perceptron(pairs: list) -> None:
    """Voiced/unvoiced binary features from string 0, N tracks → NPZ.

    Uses MIDI note events (not pitch contour) to derive voiced labels —
    a frame is voiced (1) iff it falls within a note event window, silent (0) otherwise.
    This avoids the nearest-neighbor artefact in label_frames_voiced where all frames
    snap to the nearest voiced pitch-contour observation.
    """
    from rice_Ml.preprocessing.guitarset import load_jams, get_note_events
    from rice_Ml.preprocessing.audio import load_wav, extract_string, frame_signal, frame_center_times
    from rice_Ml.preprocessing.features import extract_all
    from rice_Ml.preprocessing.dataset import label_frames_midi, save_dataset

    PERCEPTRON_OUT.parent.mkdir(parents=True, exist_ok=True)
    if PERCEPTRON_OUT.exists():
        print(f"Perceptron NPZ already exists at {PERCEPTRON_OUT.relative_to(REPO_ROOT)}, skipping.")
        return

    X_all, y_all = [], []
    subset = pairs[:N_PERCEPTRON_TRACKS]
    print(f"Extracting Perceptron features ({N_PERCEPTRON_TRACKS} tracks, voiced/unvoiced from note events) ...")
    for i, (wav, jams_path) in enumerate(subset):
        print(f"  [{i+1}/{N_PERCEPTRON_TRACKS}] {wav.name}")
        jams = load_jams(jams_path)
        note_events = get_note_events(jams, string_idx=0)
        if not note_events:
            print(f"    skipped (no note events on string 0)")
            continue
        audio, sr = load_wav(wav)
        mono = extract_string(audio, string_idx=0)
        frames = frame_signal(mono, frame_len=2048, hop_len=512)
        times = frame_center_times(frames.shape[0], frame_len=2048, hop_len=512, sr=sr)
        X = extract_all(frames, sr=sr)
        midi_y = label_frames_midi(times, note_events)
        voiced_y = (midi_y != 0).astype(np.int32)  # 1=voiced, 0=silent
        X_all.append(X)
        y_all.append(voiced_y)

    X_out = np.vstack(X_all)
    y_out = np.concatenate(y_all)
    save_dataset(X_out, y_out, PERCEPTRON_OUT)
    n0, n1 = np.sum(y_out == 0), np.sum(y_out == 1)
    print(f"  Saved → {PERCEPTRON_OUT.relative_to(REPO_ROOT)}  shape X={X_out.shape}")
    print(f"  Class balance: {n0} silent ({n0/len(y_out):.1%}), {n1} voiced ({n1/len(y_out):.1%})")


def extract_mlp(pairs: list) -> None:
    """MIDI-note multi-class features from all 6 strings, N tracks, top-K notes → NPZ."""
    from rice_Ml.preprocessing.dataset import build_multi_string_dataset, save_dataset

    MLP_OUT.parent.mkdir(parents=True, exist_ok=True)
    if MLP_OUT.exists():
        print(f"MLP NPZ already exists at {MLP_OUT.relative_to(REPO_ROOT)}, skipping.")
        return

    X_all, y_all = [], []
    subset = pairs[:N_MLP_TRACKS]
    print(f"Extracting MLP features ({N_MLP_TRACKS} tracks, all 6 strings, MIDI labels) ...")
    for i, (wav, jams) in enumerate(subset):
        print(f"  [{i+1}/{N_MLP_TRACKS}] {wav.name}")
        X, y = build_multi_string_dataset(wav, jams, label="midi")
        X_all.append(X)
        y_all.append(y.astype(np.int32))

    X_out = np.vstack(X_all)
    y_out = np.concatenate(y_all)

    # Drop silence (label 0) and keep only the TOP_N_NOTES most frequent notes
    voiced_mask = y_out != 0
    X_out, y_out = X_out[voiced_mask], y_out[voiced_mask]

    note_counts = np.bincount(y_out)
    top_notes = np.argsort(note_counts)[-TOP_N_NOTES:][::-1]
    top_mask = np.isin(y_out, top_notes)
    X_out, y_out = X_out[top_mask], y_out[top_mask]

    # Remap note IDs to contiguous 0-based labels
    note_to_class = {note: cls for cls, note in enumerate(sorted(top_notes))}
    y_out = np.array([note_to_class[n] for n in y_out], dtype=np.int32)

    save_dataset(X_out, y_out, MLP_OUT)
    print(f"  Saved → {MLP_OUT.relative_to(REPO_ROOT)}  shape X={X_out.shape}  classes={np.unique(y_out)}")
    print(f"  MIDI notes kept (sorted): {sorted(top_notes)}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    unzip_data()
    pairs = get_track_pairs()
    print(f"Found {len(pairs)} track pairs.")
    extract_perceptron(pairs)
    extract_mlp(pairs)
    print("Done.")
