import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, K, max_iter=100, eps=1e-4):
        self.K = K  # Số cụm
        self.max_iter = max_iter  # Số lần lặp tối đa
        self.eps = eps  # Ngưỡng hội tụ
        self.centroids = None  # Tâm cụm

    def init_centers(self, X):
        """
        Khởi tạo K tâm cụm ngẫu nhiên từ dữ liệu X.
        """
        return X[np.random.choice(X.shape[0], self.K, replace=False)]

    def distance(self, data, centroids):
        """
        Tính khoảng cách Euclidean giữa các điểm và các tâm cụm.
        """
        return cdist(data, centroids, metric='euclidean')

    def detect_label(self, data, centroids):
        """
        Gán nhãn cho từng điểm dựa trên tâm cụm gần nhất.
        """
        distances = self.distance(data, centroids)
        return np.argmin(distances, axis=1)

    def update_centroid(self, data, labels):
        """
        Cập nhật tâm cụm mới dựa trên trung bình các điểm trong mỗi cụm.
        """
        new_centroids = []
        for idx in range(self.K):
            points_in_cluster = data[labels == idx]
            if len(points_in_cluster) > 0:
                new_centroids.append(points_in_cluster.mean(axis=0))
            else:
                # Nếu cụm không có điểm nào, giữ nguyên tâm cụm
                new_centroids.append(self.centroids[idx])
        return np.array(new_centroids)

    def fit(self, data):
        """
        Thực hiện thuật toán K-means trên tập dữ liệu.
        """
        self.centroids = self.init_centers(data)
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            labels = self.detect_label(data, self.centroids)
            self.centroids = self.update_centroid(data, labels)

            # Kiểm tra hội tụ
            if np.linalg.norm(self.centroids - old_centroids) < self.eps:
                print(f"K-means hội tụ sau {i + 1} lần lặp.")
                break

        return labels, self.centroids
import matplotlib.pyplot as plt

# Tạo dữ liệu giả lập
np.random.seed(42)
X1 = np.random.randn(50, 2) + np.array([2, 2])
X2 = np.random.randn(50, 2) + np.array([-2, -2])
X3 = np.random.randn(50, 2) + np.array([2, -2])
X = np.vstack((X1, X2, X3))

# Áp dụng thuật toán K-means
kmeans = KMeans(K=3)
labels, centroids = kmeans.fit(X)

# Vẽ biểu đồ
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.legend()
plt.title('K-means Clustering')
plt.show()
