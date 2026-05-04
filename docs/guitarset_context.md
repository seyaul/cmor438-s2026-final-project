# GuitarSet Dataset — Full Context

This document captures everything relevant about the GuitarSet dataset as used in this project: raw structure, preprocessing pipeline, extracted features, label types, known problems, and recommendations for downstream ML tasks.

---

## 1. What GuitarSet Is

GuitarSet is a dataset of acoustic guitar recordings with frame-level annotations. It contains 360 tracks recorded by 6 players across 6 musical genres and multiple playing styles (solo/comp). Each track has:

- A **hexaphonic WAV file** — 6 channels, one per guitar string, recorded via a hex pickup with bleed between strings reduced ("debleeded"). This is the `audio_hex-pickup_debleeded` variant, referred to as `audio_hex_cln` in code.
- A **JAMS annotation file** — a JSON file containing per-string ground-truth note events and pitch contour data.

The 360 tracks break down as:
- 6 players × 6 genres × 2 styles (solo / comp) × 5 tempos/keys = 360 recordings
- Each recording is ~30 seconds

---

## 2. Track Naming Convention

Track IDs follow the pattern: `{player}_{genre}{style}-{bpm}-{key}_{type}`

Example: `00_BN1-129-Eb_solo`
- `00` — player index (0–5)
- `BN1` — genre/style code (BN = Bossa Nova, SS = Singer-Songwriter, Jazz, Rock, Funk)
- `129` — tempo in BPM
- `Eb` — key signature
- `solo` or `comp` — playing style (solo melody vs. comping/chords)

---

## 3. Guitar String Mapping

```
string_idx = 0  →  low E  (E2,  MIDI 40)
string_idx = 1  →  A      (A2,  MIDI 45)
string_idx = 2  →  D      (D3,  MIDI 50)
string_idx = 3  →  G      (G3,  MIDI 55)
string_idx = 4  →  B      (B3,  MIDI 59)
string_idx = 5  →  high E (E4,  MIDI 64)
```

Lower-indexed strings play lower-pitched notes. Strings 0 and 5 (the outermost strings) tend to be played least often, resulting in the highest silence rates.

---

## 4. Raw File Format

**Hexaphonic WAV:**
- Shape after loading: `(n_samples, 6)` — rows are time samples, columns are strings
- Sample rate: 44,100 Hz
- Normalized to float32 in `[-1.0, 1.0]`
- Each column is extracted independently as a mono signal for one string

**JAMS file (JSON):**
- Annotations array interleaves pitch_contour and note_midi: string `i` → pitch_contour at index `2*i`, note_midi at index `2*i+1`
- Indices 12–16: beat_position, tempo, chord (instructed/performed), key_mode

Two types of per-string annotation used in this project:

**Note events** (`note_midi` namespace):
```json
{"time": 1.23, "duration": 0.45, "value": 46.02}
```
`value` is a continuous MIDI float (e.g. 46.02 ≈ MIDI 46, Bb2). Onset + duration give the time range.

**Pitch contour** (`pitch_contour` namespace):
Dense time series at ~10ms resolution:
```json
{"time": [...], "value": [{"voiced": true, "frequency": 233.1, "index": 0}, ...]}
```
`voiced=True` means the string was actively vibrating at that moment.

---

## 5. Preprocessing Pipeline

### Step 1: Frame the audio

Each string's mono signal is cut into overlapping windows:

```
frame_len = 2048 samples  →  ~46.4 ms per frame at 44.1 kHz
hop_len   =  512 samples  →  frame starts every ~11.6 ms
overlap   =  75%
```

A Hann window is applied to each frame to taper the edges before FFT, reducing spectral leakage.

A 30-second recording becomes approximately **5,700 frames per string**.

Frame center timestamps are computed as:
```
times[i] = (i * hop_len + frame_len / 2) / sample_rate
```

### Step 2: Extract 18 features per frame

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `rms` | Root-mean-square energy. Near zero for silence. |
| 1 | `zcr` | Zero-crossing rate (sign changes / frame length). High for noisy/high-frequency content. |
| 2 | `centroid` | Spectral centroid in Hz — weighted mean of frequency spectrum. |
| 3 | `bandwidth` | Spectral bandwidth in Hz — weighted std dev around centroid. |
| 4 | `rolloff` | Spectral rolloff in Hz — frequency below which 85% of energy lives. |
| 5–17 | `mfcc_1`–`mfcc_13` | Mel-frequency cepstral coefficients. MFCC 1 ≈ overall energy envelope; MFCCs 2–13 capture spectral shape. Standard features for audio classification. |

**Important:** MFCCs are highly correlated with each other. This matters for algorithms sensitive to feature correlation (decision trees, Random Forest feature subsampling is especially helpful here).

All features are computed per frame using only `numpy` and `scipy`. No external audio libraries (librosa, etc.).

### Step 3: Label assignment

For each frame, the center timestamp is matched against the JAMS annotations:

**`label="midi"`** (used in MLP notebook):
- If `onset <= frame_time < offset` for any note event → label = `round(midi_note)` (integer)
- Otherwise → label = `0` (silence)
- Output dtype: `int32`

**`label="voiced"`** (used in Perceptron notebook):
- Match frame to nearest pitch contour observation
- Label = `voiced` flag from that observation (True/False)
- Output dtype: `bool`

**`label="frequency"`** (not currently used in notebooks):
- Match frame to nearest pitch contour observation
- Label = fundamental frequency in Hz (`0.0` for silence)
- Output dtype: `float64`

### Step 4: Stack across strings and tracks

`build_multi_string_dataset()` runs the pipeline for all 6 strings of one track and stacks the results. For the full dataset, results are stacked across all 360 tracks.

---

## 6. CSV Structure (the released dataset)

Two files available via GitHub Releases:
- `guitarset_subset.csv.gz` — 10 tracks, ~124K rows (~19 MB compressed)
- `guitarset_full.csv.gz` — 360 tracks, ~5.6M rows (~855 MB compressed)

Loaded via `rice_Ml.datasets.load_guitarset(subset=True/False)`. Downloaded on first call, cached at `~/.cache/rice_ml/`, SHA-256 verified.

**Column layout:**
```
track_id      string_idx   rms    zcr    centroid  bandwidth  rolloff  mfcc_1 ... mfcc_13  midi_label
str           int (0-5)    float  float  float     float      float    float  ... float    int
```

- `track_id`: recording identifier (e.g. `00_BN1-129-Eb_solo`)
- `string_idx`: which guitar string (0=low E, 5=high E)
- Features: 18 columns (rms, zcr, centroid, bandwidth, rolloff, mfcc_1–mfcc_13)
- `midi_label`: MIDI note number (0 = silence, 40–88 = active note)

**Note:** absolute frame timestamps are NOT stored in the CSV. Frames are in sequential time order within each `(track_id, string_idx)` group. To reconstruct timestamps: `frame_index * 512 / 44100`.

---

## 7. Class Imbalance — The Core Problem

### Real numbers from the 10-track subset:

| | Frames | % of total |
|--|--------|-----------|
| Silence (label=0) | 89,607 | **71.9%** |
| All voiced frames | 34,965 | 28.1% |

**Per-string silence rates:**
```
String 0 (low E):   90.1% silence
String 1 (A):       76.0% silence
String 2 (D):       58.4% silence
String 3 (G):       54.2% silence
String 4 (B):       62.9% silence
String 5 (high E):  89.9% silence
```

**Most frequent voiced notes (subset):**
```
MIDI 61 (C#5/Db5):  3,397 rows  (2.7%)
MIDI 56 (Ab4/G#4):  3,254 rows  (2.6%)
MIDI 57 (A4):       2,796 rows  (2.2%)
...
MIDI 58 (Bb4):      1,214 rows  (< 1%)
```

The most common individual note is still only ~3% of total data. Individual note classes in the full dataset are even sparser relative to silence.

### Why silence dominates:

This is not a data quality problem — it reflects real guitar music:
1. A guitar string only makes sound when actively plucked. Between notes there are physical gaps.
2. The dataset records ALL 6 strings simultaneously. At any moment, 3–5 strings are typically silent.
3. The low E and high E strings are played least often in most guitar styles, giving them ~90% silence rates.
4. Silence between phrases, rests, and string damping all contribute.

### Why this breaks naive classifiers:

- A model predicting "silence" for every frame achieves ~72% accuracy with zero actual learning.
- Gradient-based models (MLP with MSE loss) are dominated by the majority class signal.
- Each individual note class gets ~1–3% of gradient updates — not enough to learn a reliable fingerprint.
- **Especially bad with ReLU + MSE loss** (the current MLP configuration): outputs are raw activations, not probabilities, so minority class confidence scores are unreliable.
- With Softmax + CrossEntropy (not currently implemented in this codebase), the imbalance would be less catastrophic but still significant.

---

## 8. Why the MLP Performs Worse Than the Perceptron on Binary Classification

The MLP notebook trains on 13-class MIDI note classification (silence + top-12 notes) and then collapses predictions to binary (silence vs. voiced). This is inherently disadvantaged:

1. The MLP was never trained to optimize the voiced/silent boundary — it was trained to separate individual notes from each other.
2. ReLU + MSE outputs are raw activations, not calibrated probabilities — argmax is unreliable.
3. The Perceptron was trained directly for the binary task with a purpose-built loss.

Observed results on derived binary classification (MLP 13-class → collapsed):
```
Metric      MLP (derived)   Perceptron (direct)
Accuracy    0.9006          0.9622
Precision   0.7247          0.8799
Recall      0.7158          0.7146
F1          0.7202          0.7887
```

---

## 9. Recommended Approaches for Future Work

### Option A: Two-stage pipeline (most principled)
- **Stage 1:** Binary classifier (Perceptron, Logistic Regression) — voiced vs. silent
- **Stage 2:** Multi-class classifier (MLP, Random Forest) trained **only on voiced frames** — which note?
- The MLP's training set becomes 100% voiced frames, eliminating silence imbalance entirely.
- Mirrors real-world voice activity detection + recognition pipelines.

### Option B: Undersample silence at load time
```python
voiced_df = df[df['midi_label'] != 0]
silence_df = df[df['midi_label'] == 0].sample(n=len(voiced_df), random_state=42)
df_balanced = pd.concat([voiced_df, silence_df]).sample(frac=1, random_state=42)
```

### Option C: Class-weighted loss
Weight each training example by the inverse of its class frequency. Requires modifying the MLP loss function.

### For ensemble methods (Random Forest, Bagging, AdaBoost, Stacking):
- The 18-feature, multi-class setup is a good fit.
- Random Forest's feature subsampling (`sqrt(18) ≈ 4` per split) is meaningful because MFCCs are correlated.
- AdaBoost partially mitigates imbalance by upweighting misclassified minority examples each round.
- Recommended: use voiced-only frames for a clean demonstration; include silence only if intentionally studying the imbalance effect.
- 124K rows (subset) is sufficient; full 5.6M rows is overkill for a notebook demo.

---

## 10. Key Code Locations

| File | Purpose |
|------|---------|
| `src/rice_Ml/preprocessing/audio.py` | WAV loading, channel extraction, framing, Hann windowing |
| `src/rice_Ml/preprocessing/features.py` | RMS, ZCR, spectral features, MFCC, `extract_all()` |
| `src/rice_Ml/preprocessing/guitarset.py` | JAMS parsing, note events, pitch contour |
| `src/rice_Ml/preprocessing/dataset.py` | Full pipeline: `build_dataset()`, `build_multi_string_dataset()`, label functions |
| `src/rice_Ml/datasets.py` | GitHub Release loader: `load_guitarset(subset=True/False)` |
| `data/guitarset_subset.csv` | 10-track subset, local copy |
| `data/guitarset_full.csv.gz` | Full 360-track dataset, local copy |

---

## 11. Quick-Start Usage

```python
from rice_Ml.datasets import load_guitarset, FEATURE_COLS

# Load subset (10 tracks, ~124K rows)
df = load_guitarset(subset=True)

# Load full dataset (360 tracks, ~5.6M rows)
df = load_guitarset(subset=False)

# Features and labels
X = df[FEATURE_COLS].values          # shape (n, 18)
y = df['midi_label'].values           # shape (n,), 0=silence

# Voiced-only subset (recommended for note classification)
df_voiced = df[df['midi_label'] != 0]

# Single string (e.g. for Perceptron binary voiced/silent)
df_s0 = df[df['string_idx'] == 0]
y_binary = (df_s0['midi_label'] != 0).astype(int)

# Top-N note classification
top_n = df['midi_label'].value_counts().iloc[1:13].index  # skip silence at index 0
df_top = df[df['midi_label'].isin(top_n)]
```

`FEATURE_COLS = ['rms', 'zcr', 'centroid', 'bandwidth', 'rolloff', 'mfcc_1', ..., 'mfcc_13']`
