# MLP — Multi-Class MIDI Note Classification on GuitarSet

## Algorithm

The **Multi-Layer Perceptron (MLP)** extends the Perceptron with stacked nonlinear layers, enabling it to learn complex, non-linearly-separable decision boundaries (Universal Approximation Theorem, Cybenko 1989).

**Forward pass** through L layers:
```
z^(l) = W^(l) a^(l−1) + b^(l)
a^(l) = σ(z^(l))
```

**Backpropagation** applies the chain rule in reverse:
```
δ^(L) = ∇ŷ L ⊙ σ'(z^(L))
δ^(l) = (W^(l+1)ᵀ δ^(l+1)) ⊙ σ'(z^(l))
```

Weights are updated via mini-batch SGD:  `θ ← θ − η ∇θ L`

## Dataset

- **Source:** GuitarSet (`audio_hex_cln`, debleeded 6-string hex pickup)
- **Tracks used:** 10, all 6 strings
- **Frames:** 24,985 × 18 features (voiced frames only — silence removed)
- **Task:** 12-class — which MIDI note is being played?
- **Classes:** Top-12 most frequent MIDI notes, remapped to integer labels 0–11

### Features (18 per frame)

| Feature | What it measures |
|---------|-----------------|
| RMS | How loud the frame is — near zero for silence, higher for active notes |
| ZCR | Zero-crossing rate: how often the signal crosses zero — higher for noisy or silent frames |
| Spectral centroid | The "brightness" of the sound — where the energy is concentrated in the frequency spectrum |
| Spectral bandwidth | How spread out the energy is around the centroid |
| Spectral rolloff | The frequency below which 85% of the signal energy lies |
| MFCC 1–13 | Mel-frequency cepstral coefficients: a compact encoding of the overall spectral shape (timbre) |

### Label construction

Silent frames (MIDI label 0) are dropped. The top 12 most frequent MIDI notes across all tracks and strings are retained; labels are remapped to contiguous integers 0–11.

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
jupyter notebook mlp_from_scratch.ipynb
```

## Expected Outputs

- **Loss curve:** MSE decreases over 80 epochs (log scale shows rapid early improvement).
- **Test accuracy:** Depends on architecture; [64, 32] typically achieves 50–70% on 12-class task.
- **Confusion matrix:** Strong diagonal; adjacent-pitch classes show the most confusion.
- **Architecture exploration:** [128, 64, 32] achieves the lowest training loss but similar test accuracy to [64, 32] — signs of mild overfitting.

## Notes

- The MLP uses **MSE loss with one-hot targets** (categorical cross-entropy backprop through Softmax is not yet implemented in `rice_Ml`).
- Sklearn's MLPClassifier uses Adam + cross-entropy by default and will outperform our SGD + MSE baseline.
- For polyphonic transcription (multiple simultaneous notes), this single-label setup would need to be extended to multi-label classification.
