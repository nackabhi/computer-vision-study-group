# cv_learn/linear_classifiers/data_utils.py
import os, pickle, tarfile, urllib.request
import numpy as np

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def _download_and_extract(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(tgz_path):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR_URL, tgz_path)
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        print("Extracting CIFAR-10...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

def _load_batch(batch_path: str):
    with open(batch_path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        X = d[b"data"]  # (N, 3072)
        y = d[b"labels"]
        X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1)  # (N,32,32,3)
        y = np.array(y, dtype=np.int64)
    return X, y

def load_cifar10(data_dir: str, download: bool=True):
    """
    Returns:
      X_train, y_train, X_test, y_test as numpy arrays
    """
    if download: _download_and_extract(data_dir)
    base = os.path.join(data_dir, "cifar-10-batches-py")
    xs, ys = [], []
    for i in range(1, 6):
        Xi, yi = _load_batch(os.path.join(base, f"data_batch_{i}"))
        xs.append(Xi); ys.append(yi)
    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    X_test, y_test = _load_batch(os.path.join(base, "test_batch"))
    return X_train, y_train, X_test, y_test

def preprocess(X_train, X_test, num_train=None, num_val=5000, num_test=10000, dev_frac=0.1, seed=42):
    """
    - Subsample for dev if needed (dev_frac of train).
    - Mean/std normalize (channel-wise).
    - Add bias column later in model if you prefer; here we keep raw features.
    Returns dict with train/val/test/dev splits (X as (N,D) flattened).
    """
    rng = np.random.default_rng(seed)
    Ntr = X_train.shape[0] if num_train is None else min(num_train, X_train.shape[0])

    idx = rng.permutation(X_train.shape[0])[:Ntr]
    Xtr = X_train[idx]
    ytr = None  # will be passed separately in train script

    # Split val from end of chosen train set
    val = min(num_val, Xtr.shape[0]//10)  # fallback if Ntr small
    X_val = Xtr[-val:]
    X_trn = Xtr[:-val]

    # Dev subset (10% default) out of training
    dev_n = max(1, int(X_trn.shape[0]*dev_frac))
    dev_idx = rng.choice(X_trn.shape[0], size=dev_n, replace=False)
    X_dev = X_trn[dev_idx]
    X_trn_small = np.delete(X_trn, dev_idx, axis=0)

    # Test subset
    X_tst = X_test[:num_test]

    # Channel-wise mean/std over *training only*
    mean = X_trn_small.mean(axis=(0,1,2), keepdims=True)
    std = X_trn_small.std(axis=(0,1,2), keepdims=True) + 1e-7
    def norm(X): return (X - mean) / std

    X_trn_small = norm(X_trn_small).reshape(X_trn_small.shape[0], -1)
    X_val = norm(X_val).reshape(X_val.shape[0], -1)
    X_dev = norm(X_dev).reshape(X_dev.shape[0], -1)
    X_tst = norm(X_tst).reshape(X_tst.shape[0], -1)

    return {
        "X_train": X_trn_small,
        "X_val": X_val,
        "X_dev": X_dev,
        "X_test": X_tst,
        "mean": mean,
        "std": std
    }

