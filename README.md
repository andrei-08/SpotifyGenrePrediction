# Spotify Genre Classification

Predicting a track's musical genre from its audio features (danceability, energy,
loudness, tempo, valence, etc.), framed as a numerical-methods project: genre
classification is solved as a least-squares problem using the same matrix
decompositions — **Gaussian elimination, LU, and QR** — applied to ordinary linear
regression.

## Dataset

~114,000 Spotify tracks with 14 numeric audio features and a `track_genre` label.
The raw 114 genres are noisy (a mix of true genres, moods, and languages), so they
are consolidated down to ~21 clean classes:

- **Merged** related styles into umbrella genres (e.g. all metal subgenres → `metal`,
  the Japanese pop/rock/idol cluster → `j_music`, reggae/dancehall/ska → `reggae_ska`).
- **Dropped** labels that audio features cannot learn — language/nationality tags
  (`french`, `german`, …) and, optionally, mood tags (`happy`, `study`, …), since
  these describe feel or origin rather than musical style.

## Approach

The task is multiclass classification. Three models are trained and compared on the
same stratified train/test split with standardized features:

1. **Least-squares classifier (numerical-methods core).** The genres are one-hot
   encoded into an indicator matrix `Y`, and the system `X·B = Y` is solved three
   ways — Gaussian elimination and LU (via the normal equations `XᵀX·B = XᵀY`) and
   QR (factoring `X` directly). Prediction is the genre with the highest fitted score
   (`argmax`). The script also reports `cond(X)` vs `cond(XᵀX)` to show why QR is the
   most numerically reliable: forming `XᵀX` squares the condition number, while QR
   avoids it.
2. **Logistic regression** — the standard linear classifier, fit iteratively.
3. **Random forest** — a nonlinear baseline that captures curved decision boundaries
   the linear models cannot.

The comparison spans three rungs of model complexity, with the direct decomposition
methods as the foundation. As expected, the least-squares and logistic models hit a
similar linear ceiling, while the random forest clears it.

## Usage

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
python train_genre_model.py
```

Toggles at the top of `train_genre_model.py` control the genre grouping, which labels
to drop, class weighting, and which models to run.

## Outputs

- Per-model test accuracy and per-class precision/recall/F1
- Confusion matrices (`confusion_matrix_logreg.png`, `confusion_matrix_rf.png`,
  `confusion_matrix_leastsq_qr.png`)
- Top confused class pairs and the conditioning report for the three solvers
