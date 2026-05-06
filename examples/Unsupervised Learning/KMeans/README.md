# K-Means Clustering From Scratch

## Overview

This notebook implements **K-Means clustering** from scratch to discover natural groupings in GuitarSet audio data without labels.

## What You'll Learn

- **Algorithm intuition**: E-step (assignment) and M-step (centroid update) cycles
- **K-Means++ initialization**: Why random initialization is risky and how K-Means++ fixes it
- **Objective function**: Within-cluster sum of squares (WCSS) and why it's NP-hard
- **Elbow method and silhouette scores**: How to choose the number of clusters
- **Convergence criteria**: When to stop iterating

## Running the Notebook

```bash
cd examples/Unsupervised\ Learning/KMeans
jupyter notebook kmeans_from_scratch.ipynb
```

**First run:** The notebook downloads GuitarSet audio features and caches them at `~/.cache/rice_ml/`. Subsequent runs use the cache.

## Key Experiments

1. **K=2 Voiced/Silent Discovery**  
   Can K-Means separate voiced frames (note playing) from silent frames without labels?

2. **Elbow Method (K=2–15)**  
   Which value of K minimizes inertia while avoiding over-clustering?

3. **Note Discovery (K=12 on Voiced Frames)**  
   Can K-Means identify individual guitar notes? Why does it struggle?

4. **Initialization Comparison**  
   K-Means++ vs. random initialization across 10 seeds.

## Dataset

- **Source**: GuitarSet (all 6 strings, 10-track subset)
- **Frames**: ~125,000 audio frames (46 ms each)
- **Features**: 18-dimensional — see table below
- **Labels**: MIDI note number per frame (0 = silence), used only for evaluation

### Features (18 per frame)

| Feature | What it measures |
|---------|-----------------|
| RMS | How loud the frame is — near zero for silence, higher for active notes |
| ZCR | Zero-crossing rate: how often the signal crosses zero — higher for noisy or silent frames |
| Spectral centroid | The "brightness" of the sound — where the energy is concentrated in the frequency spectrum |
| Spectral bandwidth | How spread out the energy is around the centroid |
| Spectral rolloff | The frequency below which 85% of the signal energy lies |
| MFCC 1–13 | Mel-frequency cepstral coefficients: a compact encoding of the overall spectral shape (timbre) |

## Main Takeaways

✓ K-Means discovers the voiced/silent boundary well  
✓ The elbow method suggests K=2 or K=6 as natural cluster counts  
✗ K-Means cannot reliably separate individual notes (fine-grained pitch requires supervision)  
✓ K-Means++ significantly outperforms random initialization  
✓ Feature standardization is essential — without it, high-variance features dominate

## References

- Lloyd, S. (1982). Least squares quantization in PCM.
- Arthur & Vassilvitskii (2007). K-Means++: The Advantages of Careful Seeding
- Course material: CMOR 438, Rice University
