# Indices of Cluster Validity
import numpy as np
import math
from utility import norm_distances, extract_labels

# DI +
def dunn(data: np.ndarray, labels: np.ndarray) -> float:
    C = len(np.unique(labels))
    cluster_points = [data[labels == i] for i in range(C)]
    cluster_centers = np.array([np.mean(points, axis=0) for points in cluster_points])
    # Tính khoảng cách nhỏ nhất giữa các tâm cụm
    min_cluster_distance = np.inf
    from itertools import combinations
    for i, j in combinations(range(C), 2):
        dist = norm_distances(cluster_centers[i], cluster_centers[j])
        min_cluster_distance = min(min_cluster_distance, dist)
    # Tính đường kính lớn nhất của các cụm
    max_cluster_diameter = 0
    for points in cluster_points:
        if len(points) > 1:  # Cụm phải có ít nhất 2 điểm để tính đường kính
            distances = norm_distances(points[:, np.newaxis], points, axis=2)
            cluster_diameter = np.max(distances)
            max_cluster_diameter = max(max_cluster_diameter, cluster_diameter)
    # Tránh chia cho 0
    if max_cluster_diameter == 0:
        return np.inf
    return min_cluster_distance / max_cluster_diameter
    # =============================================
    # from scipy.spatial.distance import cdist, pdist, squareform
    # distances = pdist(data)
    # dist_matrix = squareform(distances)

    # labels_unique = np.unique(labels)
    # n_clusters = len(labels_unique)

    # min_inter_cluster_distance = np.inf
    # max_intra_cluster_distance = 0

    # for k in range(n_clusters):
    #     cluster_k = data[labels == k]

    #     # Tính khoảng cách lớn nhất trong cụm
    #     if len(cluster_k) > 1:
    #         max_intra_cluster_distance = max(
    #             max_intra_cluster_distance,
    #             np.max(pdist(cluster_k))
    #         )

    #     # Tính khoảng cách nhỏ nhất giữa các cụm
    #     for l in range(k + 1, n_clusters):
    #         cluster_l = data[labels == l]
    #         min_dist = np.min(cdist(cluster_k, cluster_l).flatten())
    #         min_inter_cluster_distance = min(min_inter_cluster_distance, min_dist)

    # if max_intra_cluster_distance == 0:
    #     return np.inf
    # return min_inter_cluster_distance / max_intra_cluster_distance


# DB -
def davies_bouldin(data: np.ndarray, labels: np.ndarray) -> float:
    # C = len(np.unique(labels))
    # cluster_centers = np.array([data[labels == i].mean(axis=0) for i in range(C)])

    # # Tính độ lệch chuẩn cho mỗi cụm
    # # dispersions = np.zeros(n_clusters)
    # dispersions = [np.mean(norm_distances(data[labels == i], cluster_centers[i], axis=1)) for i in range(C)]

    # result = 0
    # for i in range(C):
    #     max_ratio = 0
    #     for j in range(C):
    #         if i != j:
    #             ratio = (dispersions[i] + dispersions[j]) / norm_distances(cluster_centers[i], cluster_centers[j])
    #             max_ratio = max(max_ratio, ratio)
    #     result += max_ratio
    # return result / C
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(data, labels)


# PCI fuzzy +
def partition_coefficient(membership: np.ndarray) -> float:
    N, C = membership.shape
    _pc = np.sum(np.square(membership)) / N  # PC fuzzy
    _1dc = 1/C
    return (_pc - _1dc) / (1 - _1dc)


# CE fuzzy -
def classification_entropy(membership: np.ndarray, a: float = np.e) -> float:
    """
    CE đo lường mức độ không chắc chắn trong việc gán điểm vào các cụm, giá trị càng thấp, độ hợp lệ
    của phân cụm càng tốt. CE thường kết hợp với PC, một phân cụm tốt thường có PC cao và CE thấp,
    0 ≤ 1 − P C ≤ CE
    """
    N = membership.shape[0]

    # Tránh log(0) bằng cách thêm một epsilon nhỏ cho tất cả các phần tử
    epsilon = np.finfo(float).eps
    membership = np.clip(membership, epsilon, 1)

    # Tính tỉ lệ phần trăm điểm dữ liệu thuộc về mỗi cụm
    log_u = np.log(membership) / np.log(a)  # Chuyển đổi cơ số logarit
    return -np.sum(membership * log_u) / N


# PE fuzzy -
def partition_entropy(membership: np.ndarray) -> float:
    """
    Tính chỉ số Partition Entropy index

    Parameters
    ----------
    membership: Ma trận độ thuộc 

    Return:
    Giá trị của chỉ số Partition Entropy index
    """
    return classification_entropy(membership=membership, a=np.e)

# S fuzzy +
def separation(data: np.ndarray, membership: np.ndarray, centroids: np.ndarray, m: float = 2) -> float:
    _N, C = membership.shape
    _ut = membership.T
    numerator = 0
    for i in range(C):
        diff = data - centroids[i]
        squared_diff = np.sum(diff**2, axis=1)
        numerator += np.sum((_ut[i] ** m) * squared_diff)
    center_dists = np.sum((centroids[:, np.newaxis] - centroids) ** 2, axis=2)
    np.fill_diagonal(center_dists, np.inf)
    min_center_dist = np.min(center_dists)
    return numerator / division_by_zero(min_center_dist)


# SI +
def silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    # N = len(data)
    # silhouette_vals = np.zeros(N)
    # for i in range(N):
    #     a_i = 0
    #     b_i = np.inf
    #     for j in range(N):
    #         if i != j:
    #             distance = np.sqrt(np.sum((data[i] - data[j])**2))
    #             if labels[i] == labels[j]:
    #                 a_i += distance
    #             else:
    #                 b_i = min(b_i, distance)

    #     if np.sum(labels == labels[i]) > 1:
    #         a_i /= (np.sum(labels == labels[i]) - 1)
    #     else:
    #         a_i = 0
    #     silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
    # return np.mean(silhouette_vals)
    from sklearn.metrics import silhouette_score
    return silhouette_score(data, labels)


# AC +
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # if len(y_true) != len(y_pred):
    #     raise ValueError("Độ dài của y_true và y_pred phải giống nhau")

    # correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    # total_samples = len(y_true)
    # return correct_predictions / total_samples
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


# F1 +
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:  # binary|weighted
    # tp, fp, fn = tp_fp_fn(y_true, y_pred)
    # # Tính precision và recall
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # total = precision + recall
    # return 2 * (precision * recall) / total if total > 0 else 0
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average)


# FHV fuzzy +
def hypervolume(membership: np.ndarray, m: float = 2) -> float:
    C = membership.shape[1]
    result = 0
    for i in range(C):
        cluster_u = membership[:, i]
        n_i = np.sum(cluster_u > 0)
        if n_i > 0:
            result += np.sum(cluster_u ** m) / n_i
    return result

# XB fuzzy -
def Xie_Benie(data: np.ndarray, centroids: np.ndarray, membership: np.ndarray) -> float:
    """
    Tính chỉ số Xie-Benie index

    Parameters
    ----------
    data: dữ liệu chưa được phân cụm.
    centroids: Ma trận tâm cụm 
    membership: Ma trận độ thuộc 

    Return:
    Giá trị của chỉ số Xie-Benie index
    """
    _, C = membership.shape
    labels = extract_labels(membership)
    clusters = [data[labels == i] for i in range(C)]

    from sklearn.metrics import pairwise_distances
    S_iq = np.asanyarray([np.mean([np.linalg.norm(point - centroids[i]) for point in cluster]) for i, cluster in enumerate(clusters)])
    tu = np.sum(np.square(membership) * np.square(S_iq))
    distance = pairwise_distances(centroids)
    distance[distance == 0] = math.inf
    mau = len(data) * np.min(np.square(distance))
    return tu / mau






