# Linear Classifiers (NumPy) on CIFAR-10

A from-scratch implementation of **linear SVM (hinge)** and **Softmax (cross-entropy)** classifiers using **NumPy only**.  
Includes data loading & preprocessing, vectorized losses/gradients, SGD optimizer, numerical gradient checking, and visualization of learned class templates.

---

## Directory Structure
```
cv_learn/
└── linear_classifiers/
    ├── data_utils.py   # Get data onto disk, into RAM, and into the right numeric shape/scale
    ├── svm.py          # Multiclass SVM (hinge) loss + gradient (vectorized)
    ├── softmax.py      # Softmax + Cross-Entropy loss + gradient (vectorized)
    ├── optim.py        # SGD optimizer (+ optional momentum)
    ├── grad_check.py   # Numerical (finite-diff) gradient checker
    ├── train.py        # Training loop, val selection, test eval, weight viz, saving
README.md       		# Docs, formulas, how to run, observations
```


---

## Goals

- Build intuition for **linear decision boundaries** on images  
- Implement and compare **hinge vs. softmax** losses  
- Practice **vectorized gradients** (no Python loops over samples)  
- Train via **SGD** (with optional momentum) & **L2 regularization**  
- **Numerically verify** gradients  
- **Visualize** learned class templates as images

---

## Setup

> Python 3.10+, NumPy, Matplotlib, tqdm

```
# inside cv_learn/
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy matplotlib tqdm
```

CIFAR-10 will download automatically on first run to ./data/.

## **How to run**

### Train SVM (hinge) 
``` 
python linear_classifiers/train.py --model svm --epochs 10 --viz
```

### **Train Softmax (cross-entropy)**
```
python linear_classifiers/train.py --model svm --epochs 12 -lr 0.1 --momentum 0.9 --lr_decay 0.95 --viz
```

## **Outputs**
- `out/<model>_W.npy` — learned weights `(3072 , 10)`
- `out/<model>_weights.png` — weight "templates" visualization (if `--viz`)
- console prints : running loss , train/val acc per epoch , and 
	``` 
	Best val acc: <...> | Test acc: <...> 
	```
where __best val__ acc is the highest validation accuracy during training ; the final __Test acc__ is evaluated using those best weights

### SVM accuracy and loss graphs 
<p align = "center" >
 	<img width="420" height="380" alt="svm_accuracy" src="https://github.com/user-attachments/assets/21b8e43b-b508-4712-886f-abdff7509676" />
	<img width="420" height="380" alt="svm_loss" src="https://github.com/user-attachments/assets/9a256afd-4244-45fe-a6cd-811b377b2da0" />
</p>

### SOFTMAX accuracy and loss graphs
<p align = "center" >
	<img width="420" height="380" alt="softmax_accuracy" src="https://github.com/user-attachments/assets/aedbc5c9-81e5-487a-9782-29e19310bc55" />
	<img width="420" height="380" alt="softmax_loss" src="https://github.com/user-attachments/assets/11b1f7df-93b4-4ac5-9ca3-4f08b58ba840" />
</p>

## Common flags 
- `--epochs` , `--batch_size`
- `--lr` , `--lr_decay` , `--momentum`
- `--reg` (L2 strength)
- `--data_dir`,`--out_dir`
- `--num_val` , `--num_test`
- `--small` (use smaller training subset)
- `--viz` (save weight images)
> if you see a save error for the PNG , create the folder once : ` mkdir - p out`


## File-by-File Explanation

### `data_utils.py`
- **CIFAR-10 download/extract**
  - `_download_and_extract(data_dir)`: downloads `cifar-10-python.tar.gz` (if missing) and
    extracts it into `data_dir/cifar-10-batches-py/`.

- **Load a batch from disk**
  - `_load_batch(path)`: reads one CIFAR-10 batch with `pickle`, returns:
    - `X`: images as `(N, 32, 32, 3)`
    - `y`: labels as `int64` array of length `N`.

- **Load the full dataset**
  - `load_cifar10(data_dir, download=True)`: loads all 5 training batches and the test batch;
    returns:
    - `X_train, y_train, X_test, y_test`.

- **Preprocess & split (features only)**
  - `preprocess(X_train, X_test, num_train=None, num_val=5000, num_test=10000, dev_frac=0.1, seed=42)`:
    1. **Subsample/train split**: picks up to `num_train` images with a fixed RNG seed; then slices off
       a validation set of size `num_val`.
    2. **Dev subset**: from the remaining train, takes a `dev_frac` portion as a quick dev set.
    3. **Test subset**: takes first `num_test` test images.
    4. **Normalize** (channel-wise, using train-only stats):
       - Computes `mean` and `std` over the *training subset* (H,W,C axes),
       - Normalizes train/val/dev/test with those stats,
       - **Flattens** each image to a vector `(N, 3072)`.
---

### `svm.py`

- **Purpose:** Multiclass linear SVM (Crammer–Singer) loss **and** gradient (fully vectorized).
- **Inputs:**
  - `W ∈ ℝ^{D×C}`: weights
  - `X ∈ ℝ^{N×D}`: data
  - `y ∈ {0..C-1}^N`: labels
  - `reg`: L2 regularization strength
  - `delta`: hinge margin (default `1.0`)
- **Computations:**
  1. Scores: `S = X @ W  ∈ ℝ^{N×C}`
  2. Correct-class score per sample: `s_y = S[np.arange(N), y][:, None]`
  3. Margins: `M = max(0, S - s_y + delta)`; set `M[np.arange(N), y] = 0`
  4. **Loss:**  
    Loss = average sum of margins + L2 penalty reg * || W ||<sup>2</sup> 
  5. **Gradient:**  
     Build a mask `mask = (M > 0)`; for each row `i`, set  
     `mask[i, y[i]] = - sum_c mask[i, c]`. Then:
     dW = X <sup>T</sup> @ mask / N + 2*reg*W
- **Returns:** `(loss, dW)` where `dW` has shape `D×C`.
- **Notes:** `delta=1` is the standard hinge margin; using the mask trick avoids loops.

---

### `softmax.py`
Vectorized **Softmax + Cross-Entropy** loss and gradient:

**Purpose:** Compute the multiclass **softmax (cross-entropy) loss** and its **gradient** for a linear classifier in a fully vectorized way.

**Inputs**
- `W ∈ ℝ^{D×C}`: weights (D features, C classes)
- `X ∈ ℝ^{N×D}`: minibatch data (N samples)
- `y ∈ {0..C-1}^N`: integer class labels
- `reg`: L2 regularization strength

**Steps**
1. **Scores:** `S = X @ W`  `(N×C)`
2. **Numerical stability:** `S -= S.max(axis=1, keepdims=True)` (row-wise shift)
3. **Softmax probabilities:**  
   `P = exp(S) / exp(S).sum(axis=1, keepdims=True)` (rows sum to 1)
3. Convert to probabilities via softmax: `probs = exp(scores) / sum(exp(scores))`.

4. Loss = mean negative log-prob of the correct class + L2 penalty `reg * ||W||^2`.

5. Gradient: start with `dscores = probs`; subtract 1 at the correct class for each sample; then `dW = X^T @ dscores / N + 2 * reg * W`.

**Returns**
- `loss` (scalar), `dW ∈ ℝ^{D×C}`

**Why subtract the row max?**  
Prevents overflow in `exp`. After shifting, the largest score per row is 0, so `exp(0)=1` and others are ≤ 1.

**Notes**
- Ensure `y` is integer-typed and within `[0, C-1]`.
- If you add a bias via the **bias trick** (append a ones column to `X`), don’t introduce a separate `b`.
- Regularization helps generalization and stabilizes training.


---

### `optim.py`
* This defines a tiny stochastic gradient descent (SGD) optimizer.
* **`SGD.__init__(lr, weight_decay, momentum)`** stores hyperparameters.
    * **`weight_decay`** is not used here (regularization is already added inside the loss functions).
    * **`momentum`** adds a velocity buffer **`v`** to smooth/accelerate updates.
* **`step(W, grad)`** updates the weights:
    * If **`momentum == 0`**: plain SGD $\rightarrow$ `W -= lr * grad`.
    * If **`momentum > 0`**: momentum SGD $\rightarrow$ keep a velocity `v = momentum * v - lr * grad`, then `W += v`.

So you compute **`loss, grad = loss_fn(W, ...)`**, then call **`W = opt.step(W, grad)`** each iteration.

---

### `grad_check.py`

* It's a numerical gradient checker. You give it a function $f(\mathbf{W})$ that returns a **scalar loss** and a **weight array $\mathbf{W}$**. It estimates the gradient numerically (with finite differences) so you can compare against your analytic gradient from backprop.

#### `eval_numerical_gradient(f, W, h=1e-5)`

* Loops over every element $\mathbf{W}[\text{idx}]$.
* Perturbs that one element by **$+h$** and **$-h$**, evaluates the loss each time ($f\_pos, f\_neg$).
* Uses the **central difference formula**:

$$
\frac{\partial f}{\partial \mathbf{W}_{\text{idx}}} \approx \frac{f(\mathbf{W} + h \mathbf{e}_{\text{idx}}) - f(\mathbf{W} - h \mathbf{e}_{\text{idx}})}{2h}
$$

* Restores $\mathbf{W}[\text{idx}]$ and writes the estimate into $\text{grad}[\text{idx}]$.
* Returns a gradient array the **same shape as $\mathbf{W}$**.

#### `relative_error(a, b, eps=1e-12)`

* Computes a single number: the **max relative difference** between two arrays $a$ and $b$:

$$
\max \frac{|a - b|}{\max(1, |a| + |b| + \epsilon)}
$$

* Use it to compare **numeric grad vs analytic grad**—smaller is better (e.g., **$\sim 1\text{e-}7$ to $1\text{e-}4$** is typically OK depending on $h$ and noise).)

---

### `train.py`
#### What it trains

* A **linear classifier** on **CIFAR-10** using either:
    * Multiclass **SVM** (`--model svm`) or
    * **Softmax** (cross-entropy) (`--model softmax`)


### Data flow

1.  Load & flatten CIFAR-10 (`load_cifar10`) to shape $(N, 32*32*3)$.
2.  **Deterministic split** (`make_splits`) so $\mathbf{X}$ and $\mathbf{y}$ stay aligned:
    * Train / Val / Dev (small debug subset) / Test.
3.  **Standardize features** using train stats only (mean 0, std 1).
4.  **Bias trick** (by default): appends a column of ones to $\mathbf{X}$ so the model learns an intercept.



#### Model & loss

* Weights $\mathbf{W}$ initialized small random: shape $(D+1, C)$.
* **Pick loss**:
    * `svm_loss_vectorized` (**hinge**) or
    * `softmax_loss_vectorized` (**cross-entropy**),
    * both with **L2 regularization** included.



#### Optimization

* **SGD optimizer** (`optim.SGD`) with optional **momentum**.
* **Mini-batch training**:
    * Compute $\text{loss}, \text{d}\mathbf{W}$ on batch
    * Update $\mathbf{W} = \text{opt.step}(\mathbf{W}, \text{d}\mathbf{W})$
* **At each epoch**:
    * Compute full **train/val accuracy**
    * **Track best $\mathbf{W}$** and keep a copy of the best $\mathbf{W}$
    * Optional **learning-rate decay**



#### Logging & outputs

* **Saves best weights** to `out/<model>_W.npy`.
* **Plots and saves**:
    * `out/<model>_accuracy.png` (train & val accuracy per epoch)
    * `out/<model>_loss.png` (epoch loss curve)
* **Optional**: visualize class weights as $32 \times 32 \times 3$  plates (`--viz`) to `out/<model>_weights.png`.

---

## Key Formulas


## SVM (Hinge)

$$
m_{i,j} = \max(0, s_{i,j} - s_{i,y_i} + \Delta), \quad L = \frac{1}{N} \sum_i \sum_{j \neq y_i} m_{i,j} + \lambda \|W\|_2^2
$$

---

## Softmax + Cross-Entropy

$$
p_{i,j} = \frac{e^{s_{i,j}}}{\sum_k e^{s_{i,k}}}, \quad L = -\frac{1}{N} \sum_i \log p_{i,y_i} + \lambda \|W\|_2^2
$$



























