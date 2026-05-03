# Perceptron — Binary Note Detection on GuitarSet

## Algorithm

The **Perceptron** (Rosenblatt, 1958) is a linear binary classifier. It learns a hyperplane that separates two classes by iteratively adjusting weights for every misclassified sample:

```
w ← w − η (ŷ − y) x
b ← b − η (ŷ − y)
```

where η is the learning rate, y ∈ {−1, +1} is the true label, and ŷ is the prediction.  
The **Perceptron Convergence Theorem** guarantees halting in finite steps if the data are linearly separable.

## Dataset

- **Source:** GuitarSet (`audio_hex_cln`, debleeded 6-string hex pickup)
- **Tracks used:** 5, string 0 (low E)
- **Frames:** 9,730 × 18 features (46 ms / frame, 512-sample hop at 44.1 kHz)
- **Task:** Binary — is a guitar note being played in this frame? (0 = silent, 1 = voiced)
- **Class balance:** ~78% silent, ~22% voiced

### Features (18 per frame)

| Index | Feature |
|-------|---------|
| 0 | RMS energy |
| 1 | Zero-crossing rate |
| 2 | Spectral centroid |
| 3 | Spectral bandwidth |
| 4 | Spectral rolloff |
| 5–17 | MFCC 1–13 |

Labels are derived from MIDI note-event annotations (frames inside a note event = voiced).

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

### Generate features (first time only)

```bash
# From repo root — requires data/audio_hex-pickup_debleeded*.zip and data/annotation.zip
python scripts/extract_example_features.py
```

This creates `guitarset_features.npz` in this folder.

### Run the notebook

```bash
jupyter notebook perceptron_from_scratch.ipynb
```

Or open in VS Code / JupyterLab and run all cells.

## Expected Outputs

- **Convergence:** The model typically converges (zero training errors) within 1–3 epochs, confirming linear separability in standardised feature space.
- **Test F1:** ~0.85–0.95 on voiced/unvoiced detection.
- **PCA visualisation:** Two rough clusters in the first two principal components.
