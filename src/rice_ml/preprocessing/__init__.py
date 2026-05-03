"""
rice_ml.preprocessing
=====================

GuitarSet data loading, feature extraction, and dataset assembly.

Quick start — from local files
-------------------------------
>>> from rice_ml.preprocessing import build_dataset
>>> X, y = build_dataset("recording_hex_cln.wav", "recording.jams", string_idx=0)

Quick start — via mirdata (no local files needed)
-------------------------------------------------
>>> from rice_ml.preprocessing import get_guitarset, build_dataset_from_track, load_dataset
>>> gs = get_guitarset(download=True)           # downloads once to ~/.mirdata/guitarset/
>>> track = gs.track("03_BN3-119-G_solo")
>>> X, y = build_dataset_from_track(track, string_idx=0)

Or load a pre-built cache (run scripts/build_cache.py first):
>>> X, y = load_dataset("data/cache/03_BN3-119-G_solo.npz")
"""

from .guitarset import load_jams, get_note_events, get_pitch_contour, get_duration, get_tempo
from .audio import load_wav, extract_string, frame_signal, frame_center_times
from .features import (
    rms_energy, zero_crossing_rate, spectral_centroid,
    spectral_bandwidth, spectral_rolloff, mfcc, extract_all,
)
from .dataset import (
    label_frames_midi, label_frames_voiced, label_frames_frequency,
    build_dataset, build_multi_string_dataset, save_dataset, load_dataset,
    build_dataset_from_track, build_multi_string_dataset_from_track,
)
from .download import get_guitarset, track_ids

__all__ = [
    # guitarset annotation parsing
    "load_jams", "get_note_events", "get_pitch_contour", "get_duration", "get_tempo",
    # audio loading and framing
    "load_wav", "extract_string", "frame_signal", "frame_center_times",
    # feature extraction
    "rms_energy", "zero_crossing_rate", "spectral_centroid", "spectral_bandwidth",
    "spectral_rolloff", "mfcc", "extract_all",
    # dataset assembly — file-path based
    "label_frames_midi", "label_frames_voiced", "label_frames_frequency",
    "build_dataset", "build_multi_string_dataset",
    # dataset assembly — mirdata track based
    "build_dataset_from_track", "build_multi_string_dataset_from_track",
    # persistence
    "save_dataset", "load_dataset",
    # mirdata helpers
    "get_guitarset", "track_ids",
]
