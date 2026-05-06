# Perceptron — Binary Note Detection on GuitarSet

## Algorithm

The **Perceptron** is a linear binary classifier. It learns a hyperplane that separates two classes by iteratively adjusting weights for every misclassified sample:

```
w ← w − η (ŷ − y) x
b ← b − η (ŷ − y)
```

where η is the learning rate, y ∈ {−1, +1} is the true label, and ŷ is the prediction.  

Despite its simplicity, the perceptron laid the foundation for more complex models like the multilayered perceptron and convolutional neural nets, and plays a huge role in advancing the field of machine learning. 

In this notebook, we will build our own perceptron and demonstrate how the perceptron learns on our Guitar dataset to determine whether a string is playing or not at a certain point in time. We will also analyze the results and any limitations, discussing how the perceptron behaves on our data and when we would potentially need more complex models.

## About the Dataset

**GuitarSet** is a dataset of annotated guitar recordings. Each audio file is sliced into frames of ~46 ms. For every frame, 18 numeric features are extracted and a MIDI label is assigned (0 = silent, non-zero = the note being played).

This notebook uses a 10-track subset (2 tracks per genre: blues, funk, jazz, rock, singer-songwriter), restricted to **string 0 (low E)** only.

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

Any frame where `midi_label != 0` is **voiced** (1); all others are **silent** (0). The result is a heavily imbalanced binary classification problem — ~90% of frames are silent.