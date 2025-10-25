# cv_learn/linear_classifiers/train.py
import os, argparse
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from data_utils import load_cifar10
from svm import svm_loss_vectorized
from softmax import softmax_loss_vectorized
from optim import SGD


# ---------------------------- utilities ----------------------------

def add_bias(X: np.ndarray) -> np.ndarray:
    """Append a column of 1s to X for bias trick."""
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

def standardize_train_val_test(Xtr, Xval, Xte):
    """Standardize (mean=0, std=1) using train stats; return X*, mean, std."""
    mean = Xtr.mean(axis=0, keepdims=True)
    std  = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mean) / std
    Xval = (Xval - mean) / std
    Xte  = (Xte  - mean) / std
    return Xtr, Xval, Xte, mean, std

def visualize_weights(W, mean, std, save_path=None, with_bias=True):
    """
    W: (D(+1), C). If with_bias, last row is bias; drop it for visualization.
    mean,std are shaped (32,32,3) for nicer color scaling.
    """
    if with_bias:
        W = W[:-1, :]  # drop bias row

    C = W.shape[1]
    # CIFAR-10 dims (32x32x3)
    w_imgs = W.reshape(32, 32, 3, C).transpose(3, 0, 1, 2)  # (C,32,32,3)

    rows = 2
    cols = (C + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(2.2*cols, 2.2*rows))
    axes = np.array(axes).reshape(-1)

    for i in range(C):
        img = w_imgs[i]
        # approximate inversion of normalization for visualization only
        img = img * std + mean
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"class {i}", fontsize=9)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def accuracy(W, X, y):
    pred = (X @ W).argmax(axis=1)
    return (pred == y).mean()

def make_splits(Xtr_all, ytr_all, Xte_all, yte_all,
                num_train=None, num_val=5000, num_test=10000, dev_frac=0.1, seed=42):
    """
    Single, deterministic split so X and y stay aligned.
    Returns dict with X_train/X_val/X_dev/X_test and y_*.
    """
    rng = np.random.default_rng(seed)

    Ntr_all = Xtr_all.shape[0]
    if num_train is None:
        # use all remaining (after taking val) as train
        num_train = Ntr_all - num_val

    # 1) one permutation for the training set
    perm = rng.permutation(Ntr_all)
    idx_tv = perm[:(num_train + num_val)]
    X_tv = Xtr_all[idx_tv]
    y_tv = ytr_all[idx_tv]

    # 2) slice validation from the end
    X_val = X_tv[-num_val:]
    y_val = y_tv[-num_val:]
    X_train_full = X_tv[:-num_val]
    y_train_full = y_tv[:-num_val]

    # 3) take a dev subset from train for quick checks
    Ntr = X_train_full.shape[0]
    Ndev = max(1, int(dev_frac * Ntr))
    dev_idx = rng.choice(Ntr, size=Ndev, replace=False)
    mask = np.ones(Ntr, dtype=bool)
    mask[dev_idx] = False

    X_train = X_train_full[mask]
    y_train = y_train_full[mask]
    X_dev   = X_train_full[dev_idx]
    y_dev   = y_train_full[dev_idx]

    # 4) test slice
    X_test = Xte_all[:num_test]
    y_test = yte_all[:num_test]

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_dev=X_dev,     y_dev=y_dev,
        X_test=X_test,   y_test=y_test
    )


# ---------------------------- main train ----------------------------

def main(args):
    # 1) Load CIFAR-10 and flatten
    Xtr_all, ytr_all, Xte_all, yte_all = load_cifar10(args.data_dir, download=not args.no_download)
    Xtr_all = Xtr_all.reshape(Xtr_all.shape[0], -1).astype(np.float32)
    Xte_all = Xte_all.reshape(Xte_all.shape[0], -1).astype(np.float32)

    # 2) Deterministic split that keeps X and y aligned
    splits = make_splits(
        Xtr_all, ytr_all, Xte_all, yte_all,
        num_train=int(len(ytr_all) * 0.5) if args.small else None,
        num_val=args.num_val,
        num_test=args.num_test,
        dev_frac=0.1,
        seed=42
    )

    Xtr, ytr = splits["X_train"], splits["y_train"]
    Xval, yval = splits["X_val"], splits["y_val"]
    Xdev, ydev = splits["X_dev"], splits["y_dev"]
    Xte,  yte  = splits["X_test"], splits["y_test"]

    # 3) Standardize (critical for linear models)
    Xtr, Xval, Xte, mean_vec, std_vec = standardize_train_val_test(Xtr, Xval, Xte)
    Xdev = (Xdev - mean_vec) / std_vec

    # 4) Add bias column (bias trick)
    if not args.no_bias:
        Xtr = add_bias(Xtr)
        Xval = add_bias(Xval)
        Xdev = add_bias(Xdev)
        Xte  = add_bias(Xte)

    N, D = Xtr.shape
    C = 10

    # 5) Initialize weights
    rng = np.random.default_rng(0)
    W = 0.001 * rng.standard_normal((D, C)).astype(np.float32)

    # 6) Pick loss function
    if args.model == "svm":
        loss_fn = lambda W, X, y, reg: svm_loss_vectorized(W, X, y, reg=reg, delta=1.0)
    else:
        loss_fn = lambda W, X, y, reg: softmax_loss_vectorized(W, X, y, reg=reg)

    # 7) Optimizer
    opt = SGD(lr=args.lr, momentum=args.momentum)

    # 8) Training loop (with histories for plotting)
    best_val = -1.0
    best_W = W.copy()

    iters_per_epoch = max(1, N // args.batch_size)
    total_iters = args.epochs * iters_per_epoch

    rng_batch = np.random.default_rng(123)

    train_hist, val_hist = [], []
    loss_hist = []         # per-epoch average loss

    pbar = trange(total_iters, desc=f"Training {args.model.upper()}")
    running_loss = 0.0
    running_cnt  = 0

    for it in pbar:
        # mini-batch
        batch_idx = rng_batch.choice(N, size=args.batch_size, replace=False)
        Xb, yb = Xtr[batch_idx], ytr[batch_idx]

        loss, dW = loss_fn(W, Xb, yb, reg=args.reg)
        W = opt.step(W, dW)

        running_loss += float(loss)
        running_cnt  += 1

        # end of epoch
        if (it + 1) % iters_per_epoch == 0:
            epoch_loss = running_loss / max(1, running_cnt)
            running_loss, running_cnt = 0.0, 0

            train_acc = accuracy(W, Xtr, ytr)  # full-train accuracy (linear is cheap)
            val_acc   = accuracy(W, Xval, yval)

            train_hist.append(float(train_acc))
            val_hist.append(float(val_acc))
            loss_hist.append(epoch_loss)

            pbar.set_postfix(loss=f"{epoch_loss:.3f}", trn=f"{train_acc:.3f}", val=f"{val_acc:.3f}")

            if val_acc > best_val:
                best_val = val_acc
                best_W = W.copy()

            if args.lr_decay < 1.0:
                opt.lr *= args.lr_decay

    # keep best weights
    W = best_W
    test_acc = accuracy(W, Xte, yte)
    print(f"Best val acc: {best_val:.4f} | Test acc: {test_acc:.4f}")

    # 9) Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f"{args.model}_W.npy"), W)

    # Plot accuracy curves
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{args.model.upper()} – Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.model}_accuracy.png"), dpi=200)
    plt.close()

    # Plot loss curve
    plt.figure(figsize=(6,4))
    plt.plot(loss_hist, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{args.model.upper()} – Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.model}_loss.png"), dpi=200)
    plt.close()

    # Optional: visualize class templates (only meaningful for SVM/Softmax linear)
    if args.viz:
        mean_img = mean_vec.reshape(32, 32, 3)
        std_img  = std_vec.reshape(32, 32, 3)
        visualize_weights(
            W, mean_img, std_img,
            save_path=os.path.join(args.out_dir, f"{args.model}_weights.png"),
            with_bias=(not args.no_bias)
        )


# ---------------------------- entrypoint ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--model", choices=["svm", "softmax"], default="svm")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=5e-2)      # tune on {1e-2..1e-1}
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--momentum", type=float, default=0.0)

    parser.add_argument("--reg", type=float, default=1e-4)     # tune on {1e-4..1e-2}
    parser.add_argument("--num_val", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=10000)

    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--small", action="store_true", help="Use ~50% of train for faster runs")
    parser.add_argument("--no_bias", action="store_true", help="Do NOT add bias column (default is to add it)")
    args = parser.parse_args()
    main(args)
