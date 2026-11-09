"""
Single Layer Perceptron (SLP) untuk klasifikasi biner Iris (Setosa = 0, Versicolor = 1).

Replikasi persis dari spreadsheet SLP (Google Sheets / Excel):
  - 4 fitur (X1..X4) + bias, bobot awal semua = 0.5
  - Aktivasi sigmoid, g(z) = 1 / (1 + exp(-z))
  - Loss per sampel = (g(z) - target)^2  -> loss epoch = rata-rata (MSE)
  - Gradien  dE/dw = 2 * (g - t) * g * (1 - g) * x   (bias: x = 1)
  - Update bobot per sampel (online / stochastic), learning rate 0.1, tanpa shuffle
  - Prediksi = 1 jika g(z) > 0.5, selain itu 0
  - Data latih: 40 Setosa + 40 Versicolor (80 baris), data validasi: 10 + 10 (20 baris)
  - Validasi tiap epoch memakai bobot hasil akhir pelatihan epoch tersebut (tanpa update)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ konfigurasi
LEARNING_RATE = 0.1
EPOCHS = 5
INIT_WEIGHT = 0.5
N_TRAIN_PER_CLASS = 40
CLASSES = {"Iris-setosa": 0, "Iris-versicolor": 1}

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "iris_binary.csv"
OUT_DIR = BASE / "output"


# ------------------------------------------------------------------ data
def load_data(path=DATA_FILE):
    """Baca CSV (urutan baris sama dengan sheet 'Data': 50 Setosa lalu 50 Versicolor)."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    X = raw[:, :4].astype(float)
    y = np.array([CLASSES[s] for s in raw[:, 4]], dtype=float)
    return X, y


def split_train_val(X, y, n_train_per_class=N_TRAIN_PER_CLASS):
    """40 pertama tiap kelas -> latih, sisanya (10) -> validasi. Urutan dipertahankan."""
    tr, va = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        tr.extend(idx[:n_train_per_class])
        va.extend(idx[n_train_per_class:])
    return X[tr], y[tr], X[va], y[va]


# ------------------------------------------------------------------ model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_epoch(X, y, w, lr=LEARNING_RATE):
    """Satu epoch pelatihan online. w = [bias, teta1..teta4]. Return (w_baru, g, pred)."""
    w = w.copy()
    g_all = np.zeros(len(X))
    for i, (x, t) in enumerate(zip(X, y)):
        xb = np.concatenate(([1.0], x))
        g = sigmoid(w @ xb)                       # prediksi dgn bobot SEBELUM update
        g_all[i] = g
        grad = 2.0 * (g - t) * g * (1.0 - g) * xb  # dbias, dteta1..dteta4
        w = w - lr * grad
    return w, g_all, (g_all > 0.5).astype(int)


def evaluate(X, y, w):
    """Validasi: bobot tetap, tanpa update. Return (g, pred)."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    g = sigmoid(Xb @ w)
    return g, (g > 0.5).astype(int)


def metrics(g, pred, y):
    return float(np.mean((g - y) ** 2)), float(np.mean(pred == y))


def run():
    X, y = load_data()
    Xtr, ytr, Xva, yva = split_train_val(X, y)
    w = np.full(5, INIT_WEIGHT)
    hist = []
    for ep in range(1, EPOCHS + 1):
        w, g_tr, p_tr = train_epoch(Xtr, ytr, w)
        g_va, p_va = evaluate(Xva, yva, w)
        l_tr, a_tr = metrics(g_tr, p_tr, ytr)
        l_va, a_va = metrics(g_va, p_va, yva)
        hist.append(dict(epoch=ep, train_loss=l_tr, val_loss=l_va,
                         train_acc=a_tr, val_acc=a_va, weights=w.copy()))
    return hist


# ------------------------------------------------------------------ output
def plot(hist, key, title, ylabel, fname):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(ep, [h[f"train_{key}"] for h in hist], "o-", color="#1f5fbf", lw=2, label="Training")
    ax.plot(ep, [h[f"val_{key}"] for h in hist], "s-", color="#e0752b", lw=2, label="Validation")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ep)
    if key == "acc":
        ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    hist = run()

    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Val Loss':>10} | {'Train Acc':>9} {'Val Acc':>8}")
    print("-" * 52)
    for h in hist:
        print(f"{h['epoch']:>5} | {h['train_loss']:>10.6f} {h['val_loss']:>10.6f} | "
              f"{h['train_acc']:>9.4f} {h['val_acc']:>8.4f}")
    print("\nBobot akhir [bias, teta1..teta4]:", np.round(hist[-1]["weights"], 6))

    with open(OUT_DIR / "metrics.csv", "w") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc,bias,teta1,teta2,teta3,teta4\n")
        for h in hist:
            f.write(f"{h['epoch']},{h['train_loss']:.10f},{h['val_loss']:.10f},"
                    f"{h['train_acc']:.4f},{h['val_acc']:.4f},"
                    + ",".join(f"{v:.10f}" for v in h["weights"]) + "\n")

    plot(hist, "acc", "Accuracy - Training vs Validation (Python)", "Accuracy", "accuracy_chart.png")
    plot(hist, "loss", "Loss (MSE) - Training vs Validation (Python)", "Loss (MSE)", "loss_chart.png")
    print(f"\nChart & metrics disimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
