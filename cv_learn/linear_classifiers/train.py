# cv_learn/linear_classifiers/train.py
import os, argparse
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from data_utils import load_cifar10, preprocess
from svm import svm_loss_vectorized
from softmax import softmax_loss_vectorized
from optim import SGD

def accuracy(W, X, y):
    pred = (X @ W).argmax(axis=1)
    return (pred == y).mean()

def visualize_weights(W, mean, std, save_path=None):
    """
    W: (D,C) where D=32*32*3. Visualize each class template
    mean,std used to roughly invert normalization for nicer pics.
    """
    C = W.shape[1]
    # Drop bias if you added one elsewhere; here we assume none.
    w_imgs = W.reshape(32,32,3,C).transpose(3,0,1,2)  # (C,32,32,3)

    fig, axes = plt.subplots(2, C//2 if C%2==0 else (C//2+1), figsize=(12,5))
    axes = axes.flatten()
    for i in range(C):
        img = w_imgs[i]
        # Not a perfect inversion, but good for qualitative view:
        # bring to ~original scale
        img = img * std.squeeze() + mean.squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"class {i}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

def main(args):
    X_train_all, y_train_all, X_test_all, y_test_all = load_cifar10(args.data_dir, download=not args.no_download)

    # Use labels arrays from CIFAR for splits (same order as X)
    y_train = y_train_all
    y_test  = y_test_all

    splits = preprocess(
        X_train_all, X_test_all,
        num_train=int(len(y_train)*0.5) if args.small else None,  # example: use 50% to speed up if --small
        num_val=args.num_val,
        num_test=args.num_test,
        dev_frac=0.1,
        seed=42
    )
    # Build corresponding label splits
    N_total = len(y_train)
    Ntr_small = splits["X_train"].shape[0]
    Nval = splits["X_val"].shape[0]
    Ndev = splits["X_dev"].shape[0]

    # We reconstruct y indices deterministically to match preprocess behavior:
    # Simple approach: just slice arrays in the same order we constructed X_*.
    # For clarity here, we’ll just re-derive y via random seed again.
    # In practice, you can return y's from preprocess to avoid confusion.
    rng = np.random.default_rng(42)
    # First choose Ntr from train
    idx_all = rng.permutation(X_train_all.shape[0])[:(Ntr_small + Nval + Ndev)]
    # Split val from end, dev from remaining, rest train_small
    Xtmp = idx_all
    val_idx = Xtmp[-Nval:]
    rest = Xtmp[:-(Nval)]
    dev_idx = rng.choice(rest, size=Ndev, replace=False)
    train_small_idx = np.setdiff1d(rest, dev_idx, assume_unique=False)

    y_train_small = y_train_all[train_small_idx]
    y_val = y_train_all[val_idx]
    y_dev = y_train_all[dev_idx]
    y_test_small = y_test_all[:splits["X_test"].shape[0]]

    Xtr = splits["X_train"]; ytr = y_train_small
    Xval = splits["X_val"];   yval = y_val
    Xdev = splits["X_dev"];   ydev = y_dev
    Xte  = splits["X_test"];  yte  = y_test_small

    N, D = Xtr.shape
    C = 10
    rng = np.random.default_rng(0)
    W = 0.001 * rng.standard_normal((D, C))

    if args.model == "svm":
        loss_fn = lambda W, X, y, reg: svm_loss_vectorized(W, X, y, reg=reg, delta=1.0)
    else:
        loss_fn = lambda W, X, y, reg: softmax_loss_vectorized(W, X, y, reg=reg)

    opt = SGD(lr=args.lr, momentum=args.momentum)

    best_val = -1.0
    best_W = W.copy()

    iters_per_epoch = max(1, N // args.batch_size)
    total_iters = args.epochs * iters_per_epoch

    pbar = trange(total_iters, desc=f"Training {args.model.upper()}")
    for it in pbar:
        # mini-batch
        batch_idx = rng.choice(N, size=args.batch_size, replace=False)
        Xb, yb = Xtr[batch_idx], ytr[batch_idx]

        loss, dW = loss_fn(W, Xb, yb, reg=args.reg)
        W = opt.step(W, dW)

        if (it+1) % iters_per_epoch == 0:
            # Epoch end: evaluate
            train_acc = accuracy(W, Xtr[:min(10000, Xtr.shape[0])], ytr[:min(10000, ytr.shape[0])])  # quick estimate
            val_acc = accuracy(W, Xval, yval)
            pbar.set_postfix(loss=f"{loss:.3f}", trn=f"{train_acc:.3f}", val=f"{val_acc:.3f}")
            if val_acc > best_val:
                best_val = val_acc
                best_W = W.copy()
            # optional LR decay
            if args.lr_decay < 1.0:
                opt.lr *= args.lr_decay

    W = best_W
    test_acc = accuracy(W, Xte, yte)
    print(f"Best val acc: {best_val:.4f} | Test acc: {test_acc:.4f}")

    # before any saves
    os.makedirs(args.out_dir, exist_ok=True)

    if args.viz:
    	visualize_weights(
        	W, splits["mean"], splits["std"],
        	save_path=os.path.join(args.out_dir, f"{args.model}_weights.png")
    )	

    # save weights
    np.save(os.path.join(args.out_dir, f"{args.model}_W.npy"), W)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--model", choices=["svm","softmax"], default="svm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--num_val", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=10000)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--small", action="store_true", help="Use a smaller training subset for speed")
    args = parser.parse_args()
    main(args)
