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

| Feature | What it measures |
|---------|-----------------|
| RMS | How loud the frame is — near zero for silence, higher for active notes |
| ZCR | Zero-crossing rate: how often the signal crosses zero — higher for noisy or silent frames |
| Spectral centroid | The "brightness" of the sound — where the energy is concentrated in the frequency spectrum |
| Spectral bandwidth | How spread out the energy is around the centroid |
| Spectral rolloff | The frequency below which 85% of the signal energy lies |
| MFCC 1–13 | Mel-frequency cepstral coefficients: a compact encoding of the overall spectral shape (timbre) |
