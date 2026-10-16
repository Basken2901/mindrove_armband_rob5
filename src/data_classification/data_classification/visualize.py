import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

if "--show" not in sys.argv:
    matplotlib.use("Agg")  # save files only, no windows
import matplotlib.pyplot as plt

# One fixed color per gesture, in label order (relaxed, fisting, next gesture, ...)
GESTURE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ERROR_COLOR = "#d03b3b"
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e3df"

FEATURE_NAMES = ["MAV", "WAMP", "ZC", "SSC", "WL"]
FEATURE_LONG = ["Mean absolute value", "Willison amplitude", "Zero crossings",
                "Slope sign changes", "Waveform length"]
N_CHANNELS = 8
FS = 500

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 9,
})


def color(label):
    return GESTURE_COLORS[label % len(GESTURE_COLORS)]


def plot_raw_signals(data_path, names, out_dir, seconds=2):
    """Raw EMG: one row per channel, one column per gesture, same y-scale per row."""
    n = int(seconds * FS)
    signals = [pd.read_csv(os.path.join(data_path, "train", f"{nm}_train.csv")).to_numpy()[:n, :N_CHANNELS]
               for nm in names]
    t = np.arange(n) / FS

    fig, axes = plt.subplots(N_CHANNELS, len(names), figsize=(4 * len(names), 1.1 * N_CHANNELS),
                             sharex=True, sharey="row", squeeze=False)
    for col, (nm, sig) in enumerate(zip(names, signals)):
        axes[0, col].set_title(nm, color=INK)
        for ch in range(N_CHANNELS):
            ax = axes[ch, col]
            ax.plot(t, sig[:, ch] - sig[:, ch].mean(), color=color(col), linewidth=0.7)
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(f"ch{ch + 1}", rotation=0, ha="right", va="center")
    for col in range(len(names)):
        axes[-1, col].set_xlabel("time (s)")
    fig.suptitle(f"Raw EMG, first {seconds} s of each training recording", color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_raw_signals.png"), dpi=150)
    return fig


def plot_features(X, y, names, out_dir):
    """Average value (± std) of every feature on every channel, per gesture."""
    fig, axes = plt.subplots(len(FEATURE_NAMES), 1, figsize=(10, 2.2 * len(FEATURE_NAMES)), sharex=True)
    ch = np.arange(N_CHANNELS)
    width = 0.8 / len(names)
    for f, ax in enumerate(axes):
        cols = X[:, f * N_CHANNELS:(f + 1) * N_CHANNELS]
        for label, nm in enumerate(names):
            vals = cols[y == label]
            ax.bar(ch + (label - (len(names) - 1) / 2) * width, vals.mean(axis=0), width * 0.9,
                   yerr=vals.std(axis=0), color=color(label), label=nm,
                   error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
        ax.set_title(f"{FEATURE_LONG[f]} ({FEATURE_NAMES[f]})", loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right", ncol=len(names))
    axes[-1].set_xticks(ch, [f"ch{c + 1}" for c in ch])
    fig.suptitle("Features per channel (training windows, bar = mean, line = ±1 std)",
                 color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_features_per_channel.png"), dpi=150)
    return fig


def plot_feature_space(X_train, y_train, X_test, y_test, names, models, out_dir):
    """All 40 features squeezed into 2D with PCA. Each dot is one 200 ms window."""
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=2).fit(scaler.transform(X_train))
    P_train = pca.transform(scaler.transform(X_train))
    P_test = pca.transform(scaler.transform(X_test))
    explained = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharex=True, sharey=True)
    for ax, (model_name, model) in zip(np.atleast_1d(axes), models.items()):
        wrong = model.predict(X_test) != y_test
        for label, nm in enumerate(names):
            m = y_train == label
            ax.scatter(P_train[m, 0], P_train[m, 1], s=14, color=color(label), alpha=0.35,
                       linewidths=0, label=f"{nm} (train)")
            m = (y_test == label) & ~wrong
            ax.scatter(P_test[m, 0], P_test[m, 1], s=36, color=color(label),
                       edgecolors="white", linewidths=1.2, label=f"{nm} (test)")
        ax.scatter(P_test[wrong, 0], P_test[wrong, 1], s=70, marker="X", color=ERROR_COLOR,
                   edgecolors="white", linewidths=1, label=f"misclassified test ({wrong.sum()})")
        ax.set_title(f"{model_name}: {np.mean(~wrong) * 100:.1f}% of test windows correct", loc="left")
        ax.set_xlabel(f"PC1 ({explained[0]:.0f}% of variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0f}% of variance)")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Feature space: every window as a point (40 features reduced to 2 with PCA)",
                 color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_feature_space.png"), dpi=150)
    return fig


def plot_confusion(X_test, y_test, names, models, out_dir):
    """Rows = true gesture, columns = predicted gesture. Diagonal = correct."""
    labels = np.arange(len(names))
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4), squeeze=False)
    for ax, (model_name, model) in zip(axes[0], models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)
        pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in labels:
            for j in labels:
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center",
                        color="white" if pct[i, j] > 60 else INK, fontsize=10)
        ax.set_xticks(labels, names)
        ax.set_yticks(labels, names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"{model_name}: {np.trace(cm) / cm.sum() * 100:.1f}% accuracy", loc="left")
    fig.suptitle("Confusion matrices on the test set (count and % of each true gesture)",
                 color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_confusion_matrices.png"), dpi=150)
    return fig


def plot_predictions_over_time(X_test, y_test, names, models, out_dir, step_s=0.1):
    """Test windows in recording order: what the model said vs. the truth."""
    t = np.arange(len(y_test)) * step_s
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 2.4 * len(models)), sharex=True, squeeze=False)
    for ax, (model_name, model) in zip(axes[:, 0], models.items()):
        pred = model.predict(X_test)
        ax.step(t, y_test, where="post", color=MUTED, linewidth=2, label="true gesture")
        ok = pred == y_test
        for label in range(len(names)):
            m = ok & (pred == label)
            ax.scatter(t[m], pred[m], s=12, color=color(label), zorder=3)
        ax.scatter(t[~ok], pred[~ok], s=40, marker="X", color=ERROR_COLOR, zorder=4,
                   label=f"wrong prediction ({(~ok).sum()})")
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-0.5, len(names) - 0.5)
        ax.set_title(model_name, loc="left")
        ax.legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("test windows in order (s, one window every 0.1 s, recordings placed back to back)")
    fig.suptitle("Predictions on the test set, window by window", color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_predictions_over_time.png"), dpi=150)
    return fig


def main(args=None):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_path, "..", "data")
    out_dir = os.path.join(data_path, "plots")
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(os.path.join(data_path, "features.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    names = [str(n) for n in data["label_names"]]

    models = {}
    for name in ["LDA", "SVM"]:
        path = os.path.join(data_path, f"model_{name.lower()}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if not models:
        print("No trained models found. Run classify.py first.")
        return

    plot_raw_signals(data_path, names, out_dir)
    plot_features(X_train, y_train, names, out_dir)
    plot_feature_space(X_train, y_train, X_test, y_test, names, models, out_dir)
    plot_confusion(X_test, y_test, names, models, out_dir)
    plot_predictions_over_time(X_test, y_test, names, models, out_dir)
    print("plots saved in", os.path.abspath(out_dir))

    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()