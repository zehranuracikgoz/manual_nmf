import random
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def transpose(A):
    return list(map(list, zip(*A)))

def matmul(A, B):
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def elementwise_multiply(A, B):
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def elementwise_divide(A, B, eps=1e-9):
    return [[A[i][j] / (B[i][j] + eps) for j in range(len(A[0]))] for i in range(len(A))]

def reconstruction_error(X, W, H):
    WH = matmul(W, H)
    error = 0.0
    for i in range(len(X)):
        for j in range(len(X[0])):
            error += (X[i][j] - WH[i][j]) ** 2
    return error ** 0.5

def manual_nmf(X, n_components, n_iter=200, report_every=20):
    W = [[random.random() for _ in range(n_components)] for _ in range(len(X))]
    H = [[random.random() for _ in range(len(X[0]))] for _ in range(n_components)]

    errors = {}

    for it in range(1, n_iter + 1):
        WT = transpose(W)
        H = elementwise_multiply(H, elementwise_divide(matmul(WT, X), matmul(matmul(WT, W), H)))

        HT = transpose(H)
        W = elementwise_multiply(W, elementwise_divide(matmul(X, HT), matmul(matmul(W, H), HT)))

        if it % report_every == 0:
            errors[it] = reconstruction_error(X, W, H)

    return W, H, errors


digits = load_digits()
X = MinMaxScaler().fit_transform(digits.data)
X = X[:100]
X_list = X.tolist()

print("\n=== İTERASYON BAZLI MANUEL vs SKLEARN NMF ===")

H_manual_all = {}
H_sklearn_all = {}

for k in [5, 10, 20]:
    print(f"\n--- k = {k} ---")

    W_m, H_m, manual_errors = manual_nmf(X_list, k)
    H_manual_all[k] = H_m

    nmf = NMF(n_components=k, init="random", random_state=0, max_iter=200)
    W_s = nmf.fit_transform(X)
    H_s = nmf.components_
    H_sklearn_all[k] = H_s

    sklearn_error = np.linalg.norm(X - W_s @ H_s)

    print(f"{'Iter':>6} | {'Manual':>10} | {'Sklearn':>10} | {'Fark':>10}")
    print("-" * 45)

    for it, err_m in manual_errors.items():
        diff = abs(err_m - sklearn_error)
        print(f"{it:>6} | {err_m:>10.4f} | {sklearn_error:>10.4f} | {diff:>10.4f}")

    final_manual_error = manual_errors[max(manual_errors.keys())]
  
    print("\n>>> FINAL KARŞILAŞTIRMA <<<")
    print(f"Manuel Final Hata  : {final_manual_error:.4f}")
    print(f"Sklearn Final Hata : {sklearn_error:.4f}")
    print(f"Fark               : {abs(final_manual_error - sklearn_error):.4f}")

print("\n=== GÖRSEL ÇIKTILAR: MANUEL vs SKLEARN NMF BİLEŞENLERİ ===")
for k in [5, 10, 20]:
    fig, axes = plt.subplots(2, k, figsize=(2*k, 4))

    for i in range(k):
        axes[0, i].imshow(
            np.array(H_manual_all[k][i]).reshape(8, 8),
            cmap="gray"
        )
        axes[0, i].set_title(f"Man {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(
            H_sklearn_all[k][i].reshape(8, 8),
            cmap="gray"
        )
        axes[1, i].set_title(f"Skl {i}")
        axes[1, i].axis("off")

    plt.suptitle(f"Manuel NMF vs Sklearn NMF (k={k})")
    plt.show()
