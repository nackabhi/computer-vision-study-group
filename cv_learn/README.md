# Linear Classifiers (NumPy) on CIFAR-10

A from-scratch implementation of **linear SVM (hinge)** and **Softmax (cross-entropy)** classifiers using **NumPy only**.  
Includes data loading & preprocessing, vectorized losses/gradients, SGD optimizer, numerical gradient checking, and visualization of learned class templates.

---

## Directory Structure

cv_learn/
└─ linear_classifiers/
├─ data_utils.py # CIFAR-10 download, load, train/val/dev/test splits, mean/std normalization
├─ svm.py # Multiclass SVM (hinge) loss + vectorized gradient
├─ softmax.py # Softmax + Cross-Entropy loss + vectorized gradient
├─ optim.py # SGD optimizer (+ optional momentum)
├─ grad_check.py # Numerical (finite-difference) gradient checker
├─ train.py # Training loop, accuracy, LR decay, checkpoint-by-best-val, weight viz, saving
 README.md


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

### **Train SVM (hinge) **
``` 
python linear_classifiers/train.py --model svm --epochs 10 --viz
```

### **Train Softmax (cross-entropy)**
```
python linear_classifiers/train.py --model svm --epochs 12 -lr 0.1 --momentum 0.9 --lr_decay 0.95 --viz
```

## **Outputs**
- '(out/<model>_W.npy)' — learned weights '((3072 , 10))'
- '(out/<model>_weights.png)' — weight "templates" visualization (if '(--viz)')
- console prints : running loss , train/val acc per epoch , and 
	``` 
	Best val acc: <...> | Test acc: <...> 
	```
where __best val__ acc is the highest validation accuracy during training ; the final __Test acc__ is evaluated using those best weights

## Common flags 
- '(--epochs)' , '(--batch_size)'
- '(--lr)' , '(--lr_decay)' , '('--momentum')'
- ('--reg') (L2 strength)
- ('--data_dir'),('--out_dir')
- ('--num_val') , ('--num_test')
- ('--small') (use smaller training subset)
- ('--viz') (save weight images)
> if you see a save error for the PNG , create the folder once : (' mkdir - p out').


## File-by-File Explanation

### `data_utils.py`
- **Download & extract** CIFAR-10 if missing
- **Load** batches → `(X_train, y_train, X_test, y_test)`
- **Preprocess**:
  - Build **train / val / dev / test** splits
  - Compute **channel-wise mean/std** on **training only**, then normalize all splits
  - **Flatten** images to `(N, 3072)` for linear layers
- **Returns** normalized arrays and `mean/std` (used for nicer visualizations)

---

### `svm.py`
Vectorized **multiclass hinge (SVM)** loss and gradient:
- **Scores:** `S = X @ W`
- **Margins:** `max(0, S_j − S_yi + Δ)` for `j ≠ y_i`
- **Loss:** average margins `+ reg * ||W||²`
- **Gradient:** indicator-mask trick; add `2 * reg * W`

---

### `softmax.py`
Vectorized **Softmax + Cross-Entropy** loss and gradient:
- **Stabilized scores:** subtract row-wise max
- **Probabilities:** `exp(S) / sum(exp(S))`
- **Loss:** `-mean(log p_yi) + reg * ||W||²`
- **Gradient:** `(Xᵀ @ (probs − one_hot(y))) / N + 2 * reg * W`

---

### `optim.py`
Minimal **SGD** optimizer, optional **momentum**:
- No built-in weight decay here (**L2 is in the loss**)
- `step(W, dW)` updates and returns weights

---

### `grad_check.py`
**Finite-difference** gradient checker:
- `eval_numerical_gradient(f, W)` perturbs each element of `W` by `±h` and estimates the gradient
- `relative_error(a, b)` compares analytic vs. numerical grads (expect ~`1e-8` to `1e-6`)

---

### `train.py`
End-to-end **training**:
- Load & preprocess data
- Initialize `W ~ N(0, 0.001)`
- Pick loss (**svm** or **softmax**) and optimizer (**SGD**)
- Mini-batch training with **L2**, **LR decay**, and **best-val checkpointing**
- Report **train/val** per epoch; compute final **test** accuracy
- Optional **visualization**: reshape each class weight to `32×32×3` and save as PNG
- Save final weights to `out/`

---

## Key Formulas

### SVM (Hinge)
\[
m_{i,j} = \max(0,\ s_{i,j} - s_{i,y_i} + \Delta), \quad
L = \frac{1}{N}\sum_i \sum_{j \ne y_i} m_{i,j} + \lambda \lVert W \rVert_2^2
\]

### Softmax + Cross-Entropy
\[
p_{i,j} = \frac{e^{s_{i,j}}}{\sum_k e^{s_{i,k}}}, \quad
L = -\frac{1}{N}\sum_i \log p_{i,y_i} + \lambda \lVert W \rVert_2^2
\]

























