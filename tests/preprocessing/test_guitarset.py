"""Unit tests for rice_Ml.preprocessing.guitarset."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rice_Ml.preprocessing.guitarset import (
    load_jams,
    get_note_events,
    get_pitch_contour,
    get_duration,
    get_tempo,
)


# ---------------------------------------------------------------------------
# Minimal JAMS fixture builder
# ---------------------------------------------------------------------------

def _make_jams(
    duration: float = 10.0,
    tempo: float = 120.0,
    note_events_per_string: list[list[dict]] | None = None,
    pitch_contour_per_string: list[list[dict]] | None = None,
) -> dict:
    """Return a minimal JAMS dict with 6 pitch_contour + 6 note_midi annotations."""
    n_strings = 6
    if note_events_per_string is None:
        note_events_per_string = [[] for _ in range(n_strings)]
    if pitch_contour_per_string is None:
        pitch_contour_per_string = [[] for _ in range(n_strings)]

    annotations = []
    # pitch_contour: indices 0–5
    for s in range(n_strings):
        annotations.append({
            "namespace": "pitch_contour",
            "data": pitch_contour_per_string[s],
            "time": 0, "duration": duration, "sandbox": {}, "annotation_metadata": {},
        })
    # note_midi: indices 6–11
    for s in range(n_strings):
        annotations.append({
            "namespace": "note_midi",
            "data": note_events_per_string[s],
            "time": 0, "duration": duration, "sandbox": {}, "annotation_metadata": {},
        })
    # tempo: index 12
    annotations.append({
        "namespace": "tempo",
        "data": [{"time": 0, "duration": duration, "value": tempo, "confidence": 1.0}],
        "time": 0, "duration": duration, "sandbox": {}, "annotation_metadata": {},
    })

    return {
        "annotations": annotations,
        "file_metadata": {"title": "test", "duration": duration},
        "sandbox": {},
    }


def _jams_to_file(jams: dict) -> Path:
    """Write *jams* to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jams", delete=False, mode="w")
    json.dump(jams, tmp)
    tmp.close()
    return Path(tmp.name)


@pytest.fixture
def simple_jams_path():
    note_events = [
        [{"time": 1.0, "duration": 0.5, "value": 40.0, "confidence": None},
         {"time": 2.0, "duration": 1.0, "value": 45.5, "confidence": None}],
        *[[] for _ in range(5)],
    ]
    contour = [
        [{"time": 0.0, "duration": 0.0, "value": {"voiced": False, "index": 0, "frequency": 0.0}, "confidence": None},
         {"time": 0.5, "duration": 0.0, "value": {"voiced": True,  "index": 0, "frequency": 82.4}, "confidence": None},
         {"time": 1.0, "duration": 0.0, "value": {"voiced": True,  "index": 0, "frequency": 82.4}, "confidence": None}],
        *[[] for _ in range(5)],
    ]
    path = _jams_to_file(_make_jams(note_events_per_string=note_events,
                                     pitch_contour_per_string=contour))
    yield path
    path.unlink()


# ===========================================================================
# load_jams
# ===========================================================================

class TestLoadJams:
    def test_loads_valid_file(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        assert isinstance(jams, dict)
        assert "annotations" in jams
        assert "file_metadata" in jams

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jams(tmp_path / "missing.jams")

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.jams"
        bad.write_text("not json {{{")
        with pytest.raises(ValueError, match="parse"):
            load_jams(bad)

    def test_accepts_string_path(self, simple_jams_path):
        jams = load_jams(str(simple_jams_path))
        assert "annotations" in jams


# ===========================================================================
# get_duration / get_tempo
# ===========================================================================

class TestMetadata:
    def test_duration(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        assert get_duration(jams) == pytest.approx(10.0)

    def test_tempo(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        assert get_tempo(jams) == pytest.approx(120.0)


# ===========================================================================
# get_note_events
# ===========================================================================

class TestGetNoteEvents:
    def test_returns_list_of_tuples(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        events = get_note_events(jams, string_idx=0)
        assert isinstance(events, list)
        assert all(len(e) == 3 for e in events)

    def test_onset_offset_midi_values(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        events = get_note_events(jams, 0)
        assert events[0] == pytest.approx((1.0, 1.5, 40.0))
        assert events[1] == pytest.approx((2.0, 3.0, 45.5))

    def test_sorted_by_onset(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        events = get_note_events(jams, 0)
        onsets = [e[0] for e in events]
        assert onsets == sorted(onsets)

    def test_empty_string(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        assert get_note_events(jams, string_idx=1) == []

    def test_invalid_string_idx_high(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        with pytest.raises(ValueError, match="string_idx"):
            get_note_events(jams, 6)

    def test_invalid_string_idx_negative(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        with pytest.raises(ValueError):
            get_note_events(jams, -1)

    def test_invalid_string_idx_type(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        with pytest.raises(TypeError):
            get_note_events(jams, 0.5)


# ===========================================================================
# get_pitch_contour
# ===========================================================================

class TestGetPitchContour:
    def test_returns_three_arrays(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        times, freqs, voiced = get_pitch_contour(jams, 0)
        assert isinstance(times, np.ndarray)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(voiced, np.ndarray)

    def test_shapes_match(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        times, freqs, voiced = get_pitch_contour(jams, 0)
        assert times.shape == freqs.shape == voiced.shape

    def test_voiced_is_bool(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        _, _, voiced = get_pitch_contour(jams, 0)
        assert voiced.dtype == bool

    def test_unvoiced_frequency_is_zero(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        times, freqs, voiced = get_pitch_contour(jams, 0)
        # first point is unvoiced
        assert not voiced[0]
        assert freqs[0] == 0.0

    def test_voiced_frequency_nonzero(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        times, freqs, voiced = get_pitch_contour(jams, 0)
        assert voiced[1]
        assert freqs[1] == pytest.approx(82.4)

    def test_invalid_string_idx(self, simple_jams_path):
        jams = load_jams(simple_jams_path)
        with pytest.raises(ValueError):
            get_pitch_contour(jams, 7)
