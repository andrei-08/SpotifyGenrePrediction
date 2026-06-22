from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")

#File paths
_HERE = Path(__file__).parent
DATA_PATH = _HERE.parent / "dataset.csv"
CM_PATH   = _HERE / "confusion_matrix.png"

# Columuns to keep
FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "key", "mode", "time_signature", "explicit",
]

# Columns to drop
DROP_COLS = [
    "Unnamed: 0", "track_id", "artists", "album_name", "track_name", "popularity",
]

# ── configuration toggles ─────────────────────────────────────────────────────
# Mood / activity labels (happy, sad, study, ...). They score well, but they track
# audio "feel", not musical style, so they aren't genres. Set False to keep them.
DROP_MOOD = True

# Western-language / nationality labels. A French pop song and a German pop song have
# (near-)identical audio features, so these are unlearnable noise. Always dropped.
LANGUAGE_LABELS = ["french", "german", "swedish", "british", "malay"]
MOOD_LABELS     = ["happy", "sad", "chill", "party", "romance", "study", "sleep"]

# Rows whose RAW track_genre is in this set are removed from the data entirely.
DROP_GENRES = set(LANGUAGE_LABELS) | (set(MOOD_LABELS) if DROP_MOOD else set())

# class_weight passed to both models:
#   None       -> maximize OVERALL accuracy; large classes dominate   [current goal]
#   "balanced" -> equalize per-class importance; helps rare classes, lowers accuracy
CLASS_WEIGHT = None

# Train a RandomForest alongside logistic regression for a head-to-head comparison.
ADD_RANDOM_FOREST = True

# Train the indicator-matrix least-squares classifier, solved three ways
# (Gaussian elimination, LU, QR) — the numerical-analysis core of the project.
ADD_LEAST_SQUARES = True

# ── genre mapping: raw Spotify label → umbrella class ─────────────────────────
# Any raw genre absent from this dict keeps its original name as its own class.
GENRE_MAP: dict[str, str] = {
    # metal
    "black-metal": "metal", "death-metal": "metal", "heavy-metal": "metal",
    "metalcore":   "metal", "grindcore":   "metal", "metal":       "metal",
    "hardcore":    "metal",
    # rock
    "alt-rock":    "rock",  "alternative": "rock",  "hard-rock":   "rock",
    "psych-rock":  "rock",  "punk-rock":   "rock",  "punk":        "rock",
    "grunge":      "rock",  "rock":        "rock",  "rock-n-roll": "rock",
    "rockabilly":  "rock",  "emo":         "rock",
    # electronic
    "house":             "electronic", "deep-house":       "electronic",
    "chicago-house":     "electronic", "progressive-house":"electronic",
    "techno":            "electronic", "detroit-techno":   "electronic",
    "minimal-techno":    "electronic", "trance":           "electronic",
    "edm":               "electronic", "electro":          "electronic",
    "electronic":        "electronic", "dubstep":          "electronic",
    "drum-and-bass":     "electronic", "hardstyle":        "electronic",
    "idm":               "electronic", "breakbeat":        "electronic",
    "garage":            "electronic", "dub":              "electronic",
    "club":              "electronic", "dance":            "electronic",
    # jazz_soul
    "jazz":   "jazz_soul", "blues":  "jazz_soul", "soul":   "jazz_soul",
    "funk":   "jazz_soul", "gospel": "jazz_soul", "groove": "jazz_soul",
    # latin
    "salsa":     "latin", "samba":    "latin", "sertanejo": "latin",
    "reggaeton": "latin", "latino":   "latin", "latin":     "latin",
    "mpb":       "latin", "pagode":   "latin", "forro":     "latin",
    "tango":     "latin", "spanish":  "latin",
    # hip_hop
    "hip-hop": "hip_hop", "r-n-b": "hip_hop", "trip-hop": "hip_hop",
    # pop
    "pop":      "pop", "indie-pop": "pop", "synth-pop":  "pop",
    "power-pop":"pop", "pop-film":  "pop", "k-pop":      "pop",
    "j-pop":    "pop", "cantopop":  "pop", "mandopop":   "pop",
    # folk / acoustic
    "acoustic":          "folk_acoustic", "folk":       "folk_acoustic",
    "bluegrass":         "folk_acoustic", "country":    "folk_acoustic",
    "singer-songwriter": "folk_acoustic", "songwriter": "folk_acoustic",
    "guitar":            "folk_acoustic", "honky-tonk": "folk_acoustic",
    # classical / cinematic
    "classical":  "classical", "opera":      "classical",
    "new-age":    "classical", "ambient":    "classical",
    "piano":      "classical", "show-tunes": "classical",
    # ── iteration 2: finished merges (driven by the confusion matrix) ─────────
    "children": "kids",                       # kids & children: same audience
    "j-dance": "j_music", "j-idol": "j_music",  # Japanese pop/idol/rock/anime
    "j-rock":  "j_music", "anime":  "j_music",  #   cluster (they swap constantly)
    "reggae": "reggae_ska", "dancehall": "reggae_ska",  # Caribbean / offbeat family
    "ska":    "reggae_ska",
    "brazil": "latin",                        # Brazilian music -> Latin styles
    "disney": "classical",                    # orchestral / show-tune / cinematic
    "indie":  "rock",                         # plain indie = indie rock
}                                             #   (indie-pop already maps to pop)


# 1. LOAD & CLEAN
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"RAW SHAPE : {df.shape}")
    print("\nDTYPES:")
    print(df.dtypes.to_string())

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    print("\nNULL COUNTS:")
    print(null_counts.to_string() if not null_counts.empty else "  none")

    # Drop identifier / non-predictive columns
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Remove fully-duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"\nDuplicate rows dropped : {n_before - len(df)}")

    # Remove rows with nulls in features or target
    needed = FEATURE_COLS + ["track_genre"]
    n_before = len(df)
    df = df.dropna(subset=needed)
    print(f"Null-feature rows dropped : {n_before - len(df)}")
    print(f"Working rows : {len(df)}")

    # Coerce `explicit` (may arrive as string "True"/"False")
    if df["explicit"].dtype == object:
        df["explicit"] = df["explicit"].map(
            {"True": 1, "False": 0, True: 1, False: 0}
        ).astype(int)
    else:
        df["explicit"] = df["explicit"].astype(int)

    return df


# 2. GENRE GROUPING
def apply_genre_grouping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove non-genre labels (language/nationality, and moods if DROP_MOOD)
    if DROP_GENRES:
        n_before = len(df)
        df = df[~df["track_genre"].isin(DROP_GENRES)].copy()
        print(f"\nDropped {n_before - len(df):,} rows from "
              f"{len(DROP_GENRES)} non-genre labels: {sorted(DROP_GENRES)}")

    # Unmapped genres fall back to their original raw label (kept as own class)
    df["genre_group"] = df["track_genre"].map(GENRE_MAP).fillna(df["track_genre"])

    raw_n     = df["track_genre"].nunique()
    grouped_n = df["genre_group"].nunique()

    print(f"\n{'='*60}")
    print(f"Raw genre count    : {raw_n}")
    print(f"Grouped class count: {grouped_n}")
    print("\nClass distribution (grouped):")
    dist = df["genre_group"].value_counts()
    print(dist.to_string())

    unmapped = sorted(set(df["track_genre"]) - set(GENRE_MAP))
    if unmapped:
        print(f"\nUnmapped genres kept as own class ({len(unmapped)}): {unmapped}")

    return df


# 3. BUILD X / y
def build_Xy(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    X  = df[FEATURE_COLS].astype(float).values
    le = LabelEncoder()
    y  = le.fit_transform(df["genre_group"])
    return X, y, le


# 4 & 5. SPLIT + SCALE
def split_and_scale(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"\nTrain size : {len(X_train):,}   Test size : {len(X_test):,}")
    return X_train, X_test, y_train, y_test, scaler


# 6. FIT MODEL
def train_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    print(f"\n{'='*60}")
    print("Fitting LogisticRegression (saga, balanced weights) ...")

    # saga handles large datasets well; multi_class removed in sklearn ≥ 1.7
    try:
        model = LogisticRegression(
            multi_class="multinomial",
            solver="saga",
            max_iter=2000,
            class_weight=CLASS_WEIGHT,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
    except TypeError:
        # sklearn ≥ 1.7: multi_class parameter was removed
        model = LogisticRegression(
            solver="saga",
            max_iter=2000,
            class_weight=CLASS_WEIGHT,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    print("Done.")
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray):
    """Nonlinear baseline. Trees capture curved genre boundaries that a linear
    model cannot, and they need no feature scaling (the scaled input is harmless)."""
    from sklearn.ensemble import RandomForestClassifier

    print(f"\n{'='*60}")
    print("Fitting RandomForestClassifier ...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight=CLASS_WEIGHT,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Done.")
    return model


# 6b. LEAST-SQUARES CLASSIFIER VIA DIRECT NUMERICAL METHODS

def _gaussian_elimination(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B by Gaussian elimination with partial pivoting (from scratch).
    Handles multiple right-hand sides (B has one column per genre)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    M = np.hstack([A.copy(), B.copy()])          # augmented [A | B]
    for col in range(n):
        piv = np.argmax(np.abs(M[col:, col])) + col   # partial pivot
        if piv != col:
            M[[col, piv]] = M[[piv, col]]
        pivot = M[col, col]
        for row in range(col + 1, n):                  # eliminate below pivot
            M[row, col:] -= (M[row, col] / pivot) * M[col, col:]
    X = np.zeros((n, B.shape[1]))                       # back-substitution
    for col in range(n - 1, -1, -1):
        X[col] = (M[col, n:] - M[col, col + 1:n] @ X[col + 1:]) / M[col, col]
    return X


def _lu_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B via LU factorization: A = P L U, then forward/back-substitute."""
    from scipy.linalg import lu, solve_triangular
    P, L, U = lu(A)
    Z = solve_triangular(L, P.T @ B, lower=True, unit_diagonal=True)  # L Z = P^T B
    return solve_triangular(U, Z, lower=False)                        # U X = Z


def _qr_solve(X_aug: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Solve min ||X_aug B - Y|| via QR of the TALL matrix X_aug = Q R, then
    R B = Q^T Y. Never forms X^T X, so it avoids squaring the condition number."""
    from scipy.linalg import solve_triangular
    Q, R = np.linalg.qr(X_aug)                    # reduced QR
    return solve_triangular(R, Q.T @ Y, lower=False)


class LeastSquaresClassifier:
    """Indicator-matrix least-squares classifier solved by one direct method."""

    def __init__(self, method: str = "qr"):
        self.method = method
        self.B = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LeastSquaresClassifier":
        n = X.shape[0]
        k = int(y.max()) + 1
        Y = np.zeros((n, k))                      # one-hot indicator matrix
        Y[np.arange(n), y] = 1.0
        X_aug = np.hstack([np.ones((n, 1)), X])   # intercept column

        if self.method == "qr":
            self.B = _qr_solve(X_aug, Y)
        else:                                     # normal equations route
            A   = X_aug.T @ X_aug
            rhs = X_aug.T @ Y
            if self.method == "gaussian":
                self.B = _gaussian_elimination(A, rhs)
            elif self.method == "lu":
                self.B = _lu_solve(A, rhs)
            else:
                raise ValueError(f"unknown method: {self.method}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.argmax(X_aug @ self.B, axis=1)  # highest score wins


def train_least_squares_methods(X_train, y_train) -> dict:
    """Fit the classifier three ways and report the conditioning that explains
    why QR is the most numerically reliable of the three."""
    X_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    cond_X   = np.linalg.cond(X_aug)
    cond_AtA = np.linalg.cond(X_aug.T @ X_aug)

    print(f"\n{'='*60}")
    print("NUMERICAL CONDITIONING (the reason the three methods can differ)")
    print(f"  cond(X)      = {cond_X:12,.2f}")
    print(f"  cond(X^T X)  = {cond_AtA:12,.2f}   (~ cond(X)^2)")
    print("  Gaussian/LU solve the X^T X system -> inherit the SQUARED condition")
    print("  number. QR factors X directly -> keeps the smaller cond(X). On well-")
    print("  conditioned features all three agree; the gap widens as features")
    print("  become collinear (e.g. energy & loudness).")

    print(f"\n{'='*60}")
    print("Fitting least-squares classifier: Gaussian, LU, QR ...")
    models = {m: LeastSquaresClassifier(m).fit(X_train, y_train)
              for m in ("gaussian", "lu", "qr")}

    B_g, B_lu, B_qr = models["gaussian"].B, models["lu"].B, models["qr"].B
    print(f"  ||B_gauss - B_lu|| = {np.linalg.norm(B_g - B_lu):.2e}  "
          "(same system, essentially identical)")
    print(f"  ||B_gauss - B_qr|| = {np.linalg.norm(B_g - B_qr):.2e}  "
          "(different route via X)")
    print("Done.")
    return models


# 7. EVALUATE
def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    cm_path: Path = CM_PATH,
    model_name: str = "model",
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    y_pred      = model.predict(X_test)
    class_names = le.classes_

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"[{model_name}] Test accuracy : {acc:.4f}")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print("\nClassification Report:\n", report_str)

    # ── confusion matrix plot (normalised by true count) ─────────────────────
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n        = len(class_names)
    cell_px  = max(40, 420 // n)          # shrink cells for many classes
    fig_in   = n * cell_px / 100.0
    font_sz  = max(5, min(9, cell_px // 5))

    fig, ax = plt.subplots(figsize=(fig_in * 1.1, fig_in))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        linewidths=0.3,
        linecolor="#cccccc",
        ax=ax,
        annot_kws={"size": font_sz},
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.set_title(f"Normalised Confusion Matrix — {model_name} (row = true class)",
                 fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=font_sz)
    plt.yticks(rotation=0,              fontsize=font_sz)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {cm_path}")

    return cm, class_names, acc, report_dict


# 8. TOP CONFUSED PAIRS
def print_top_confused(
    cm: np.ndarray,
    class_names: np.ndarray,
    top_n: int = 10,
) -> None:
    off_diag = cm.astype(float).copy()
    np.fill_diagonal(off_diag, 0)

    # Flatten, sort descending, take top_n indices
    flat_sorted = np.argsort(off_diag, axis=None)[::-1]
    rows, cols  = np.unravel_index(flat_sorted, off_diag.shape)

    print(f"\n{'='*60}")
    print(f"Top {top_n} most-confused pairs  (true -> predicted):")
    print(f"  {'True':<20} {'Predicted':<20} {'Count':>7}  {'% of true':>10}")
    print("  " + "-" * 60)
    for i in range(min(top_n, len(rows))):
        r, c = rows[i], cols[i]
        count = int(off_diag[r, c])
        if count == 0:
            break
        true_total   = cm[r].sum()
        pct          = 100 * count / true_total if true_total else 0
        print(f"  {class_names[r]:<20} {class_names[c]:<20} {count:>7}   {pct:>8.1f}%")


# WEAKEST CLASSES HELPER
def weakest_classes(
    report_dict: dict,
    class_names: np.ndarray,
    n: int = 3,
) -> list[tuple[str, float]]:
    rows = [
        (cls, report_dict[cls]["f1-score"])
        for cls in class_names
        if cls in report_dict
    ]
    rows.sort(key=lambda x: x[1])
    print(f"\nWeakest {n} classes by F1:")
    for cls, f1 in rows[:n]:
        print(f"  {cls:<22} F1 = {f1:.3f}")
    return rows[:n]


# MAIN
def main() -> None:
    df = load_and_clean(DATA_PATH)
    df = apply_genre_grouping(df)

    X, y, le                  = build_Xy(df)
    X_tr, X_te, y_tr, y_te, _ = split_and_scale(X, y)

    results: dict[str, float] = {}

    # ── Logistic regression (linear) ────────────────────────────────────────
    logreg = train_model(X_tr, y_tr)
    cm, class_names, acc_lr, rpt = evaluate(
        logreg, X_te, y_te, le,
        cm_path=_HERE / "confusion_matrix_logreg.png",
        model_name="LogisticRegression",
    )
    print_top_confused(cm, class_names, top_n=10)
    weakest_classes(rpt, class_names, n=3)
    results["LogisticRegression"] = acc_lr

    # ── Random forest (nonlinear) ───────────────────────────────────────────
    if ADD_RANDOM_FOREST:
        rf = train_random_forest(X_tr, y_tr)
        cm_rf, _, acc_rf, rpt_rf = evaluate(
            rf, X_te, y_te, le,
            cm_path=_HERE / "confusion_matrix_rf.png",
            model_name="RandomForest",
        )
        print_top_confused(cm_rf, class_names, top_n=10)
        weakest_classes(rpt_rf, class_names, n=3)
        results["RandomForest"] = acc_rf

    # ── Least-squares classifier via Gaussian / LU / QR (numerical methods) ──
    if ADD_LEAST_SQUARES:
        ls_models = train_least_squares_methods(X_tr, y_tr)
        # Gaussian & LU solve the same system -> report their accuracy directly
        for method in ("gaussian", "lu"):
            acc = accuracy_score(y_te, ls_models[method].predict(X_te))
            results[f"LeastSq-{method}"] = acc
        # QR gets the full treatment (report + its own confusion matrix)
        cm_ls, _, acc_qr, rpt_ls = evaluate(
            ls_models["qr"], X_te, y_te, le,
            cm_path=_HERE / "confusion_matrix_leastsq_qr.png",
            model_name="LeastSquares-QR",
        )
        print_top_confused(cm_ls, class_names, top_n=10)
        weakest_classes(rpt_ls, class_names, n=3)
        results["LeastSq-qr"] = acc_qr

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"  Classes : {len(class_names)}")
    for name, acc in results.items():
        print(f"  {name:<20} accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()