import re
import numpy as np
COLORS = ['Blue', 'Orange', 'Green', 'Red', 'Cyan', 'Yellow', 'Purple', 'Pink', 'Brown', 'Black', 'Gray', 'Beige', 'Turquoise', 'Silver', 'Gold']


# =======================================
# Làm tròn số
def round_float(number: float, n: int = 3) -> float:
    if n == 0:
        return int(number)
    return round(number, n)

# Ma trận độ thuộc ra nhãn (giải mờ)
def extract_labels(membership: np.ndarray) -> np.ndarray:
    return np.argmax(membership, axis=1)

# Chia các điểm vào các cụm
def extract_clusters(data: np.ndarray, labels: np.ndarray, n_clusters: int = 0) -> list:
    if n_clusters == 0:
        n_clusters = np.unique(labels)
    return [data[labels == i] for i in range(n_clusters)]

def norm_distances(A: np.ndarray, B: np.ndarray, axis: int = None) -> float:
    # np.sqrt(np.sum((np.asarray(A) - np.asarray(B)) ** 2))
    # np.sum(np.abs(np.array(A) - np.array(B)))
    return np.linalg.norm(A - B, axis=axis)

# Ma trận khoảng cách Euclide giữa các điểm trong 2 tập hợp dữ liệu
def distance_cdist(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    # return distance_euclidean(X,Y) if metric=='euclidean' else distance_chebyshev(X,Y)
    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric=metric)


# Khoảng cách của 2 cặp điểm trong một ma trận
def distance_pdist(data: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    from scipy.spatial.distance import pdist
    return pdist(data, metric=metric)


# Chia hết cho 0
def division_by_zero(data: np.ndarray) -> np.ndarray:
    data[data == 0] = np.finfo(float).eps
    return data
