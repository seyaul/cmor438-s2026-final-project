# Ensemble Methods — Guitar String Identification on GuitarSet

## Overview

This notebook walks through four ensemble strategies, each building on the previous:

| Method | Core idea |
|--------|-----------|
| **Bagging** | Majority vote over trees trained on bootstrap samples |
| **Random Forest** | Bagging + random feature subsets to decorrelate trees |
| **AdaBoost** | Sequential stumps, each correcting its predecessor's mistakes |
| **Stacking** | Meta-learner trained on held-out base-model predictions |

## Dataset

- **Source:** GuitarSet (`audio_hex_cln`, full 360-track dataset)
- **Task:** 6-class guitar string identification — given 18 audio features from a voiced frame, predict which string (0–5) is being played
- **Frames:** voiced frames only (midi_label ≠ 0), all 6 strings
- **Features (18 per frame):**

| Index | Feature |
|-------|---------|
| 0 | RMS energy |
| 1 | Zero-crossing rate |
| 2 | Spectral centroid |
| 3 | Spectral bandwidth |
| 4 | Spectral rolloff |
| 5–17 | MFCC 1–13 |

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

### Run the notebook

```bash
jupyter notebook ensembles_from_scratch.ipynb
```

Or open in VS Code / JupyterLab and run all cells.

## Expected Outputs

- **Decision tree baseline:** ~60–70% accuracy (one tree, no ensemble)
- **Bagging:** improvement over baseline by variance reduction
- **Random Forest:** highest accuracy among tree-based methods (~80–90%)
- **AdaBoost:** competitive, especially on hard-to-classify frames
- **Stacking:** combines RF + AdaBoost predictions via logistic regression meta-learner
- **Comparison table:** all models side-by-side with accuracy, per-class F1, and train time
