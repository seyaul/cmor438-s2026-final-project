"""
rice_Ml.preprocessing
=====================

GuitarSet data loading, feature extraction, and dataset assembly.

Quick start
-----------
>>> from rice_Ml.preprocessing.dataset import build_dataset
>>> X, y = build_dataset("recording_hex_cln.wav", "recording.jams", string_idx=0)
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
)

__all__ = [
    "load_jams", "get_note_events", "get_pitch_contour", "get_duration", "get_tempo",
    "load_wav", "extract_string", "frame_signal", "frame_center_times",
    "rms_energy", "zero_crossing_rate", "spectral_centroid", "spectral_bandwidth",
    "spectral_rolloff", "mfcc", "extract_all",
    "label_frames_midi", "label_frames_voiced", "label_frames_frequency",
    "build_dataset", "build_multi_string_dataset", "save_dataset", "load_dataset",
]
