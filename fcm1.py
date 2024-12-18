import random

class FCM:
    def __init__(self, data, n_clusters, m=2, max_iter=100, epsilon=1e-5):
        self.data = data  # Danh sách dữ liệu
        self.n_clusters = n_clusters  # Số cụm
        self.m = m  # Chỉ số mờ
        self.max_iter = max_iter  # Số lần lặp tối đa
        self.epsilon = epsilon  # Sai số hội tụ
        self.n_data = len(data)  # Số điểm dữ liệu
        self.u = self.ktmttv()  # Ma trận thành viên ban đầu
        self.centroids = [0] * self.n_clusters  # Tâm cụm ban đầu

    def ktmttv(self):
        """Khởi tạo ma trận thành viên"""
        random.seed(42)  # Đồng bộ hóa seed
        u = []
        for _ in range(self.n_data):
            row = [random.random() for _ in range(self.n_clusters)]
            s = sum(row)  # Tính tổng các phần tử trong hàng
            u.append([x / s for x in row])  # Chuẩn hóa
        return u

    def capnhat_tamcum(self):
        """Cập nhật tâm cụm"""
        for j in range(self.n_clusters):
            numerator = 0
            denominator = 0
            for i in range(self.n_data):
                u_m = self.u[i][j] ** self.m
                numerator += u_m * self.data[i]
                denominator += u_m
            self.centroids[j] = numerator / denominator if denominator != 0 else 0

    def khoang_cach(self):
        """Tính khoảng cách từ điểm dữ liệu đến các tâm cụm"""
        kcach = []
        for i in range(self.n_data):
            row = []
            for j in range(self.n_clusters):
                row.append(abs(self.data[i] - self.centroids[j]))
            kcach.append(row)
        return kcach

    def capnhat_mttv(self):
        """Cập nhật ma trận thành viên"""
        kc = self.khoang_cach()  # Lấy khoảng cách giữa các điểm và tâm cụm
        for i in range(self.n_data):
            for j in range(self.n_clusters):
                denominator = 0
                for k in range(self.n_clusters):
                    denominator += (kc[i][j] / kc[i][k]) ** (2 / (self.m - 1))
                self.u[i][j] = 1.0 / denominator

    def sai_so(self, old_u):
        """Tính sai số giữa ma trận thành viên cũ và mới"""
        ss = 0
        for i in range(self.n_data):
            for j in range(self.n_clusters):
                ss += abs(self.u[i][j] - old_u[i][j])
        return ss

    def fcm(self):
        """Thực hiện thuật toán FCM"""
        for _ in range(self.max_iter):
            old_u = [row[:] for row in self.u]  # Tạo bản sao của ma trận thành viên
            self.capnhat_tamcum()  # Cập nhật tâm cụm
            self.capnhat_mttv()  # Cập nhật ma trận thành viên
            if self.sai_so(old_u) < self.epsilon:  # Kiểm tra hội tụ
                break
        return self.u, self.centroids


# Dữ liệu đầu vào
data = [1, 3, 5, 7, 9]
n_clusters = 2

# Thực hiện FCM
fcm = FCM(data, n_clusters)
u, centroids = fcm.fcm()

print("Ma trận thành viên (u):")
for row in u:
    print(row)

print("Tâm cụm:")
print(centroids)
