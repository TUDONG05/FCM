import numpy as np


class FCM:
    def __init__(self, x, n_clusters, m=2, max_iter=100, epsilon=1e-5):
        self.x = np.array(x, dtype=np.float64).reshape(-1, 1) # các điểm dữ liệu
        self.n_clusters = n_clusters  # số cụm
        self.m = m  # chỉ số mờ
        self.max_iter = max_iter # số lần lặp tối  đa
        self.epsilon = epsilon  # sai số epsilon
        self.n_data = self.x.shape[0] #số điểm dữ liệu
        self.u = self._ktmttv() #ma trận thành viên
        self.centroids = np.zeros((self.n_clusters,1))# tâm cụm
    def _ktmttv(self):
        """Khởi tạo ma trận thành viên """
        np.random.seed(42)
        u =  np.random.rand(self.n_data,self.n_clusters)
        u=u / np.sum(u,axis=1, keepdims=True)
        # u chia cho tổng các mức độ phụ thuộc thành phần để chuẩn hóa đảm bảo tổng các phần từ cùng 1 hàng bằng 1
        return u
    
    def _capnhat_tamcum(self):
        """cập nhật tâm cụm """
        new_u=self.u**self.m
        self.centroids = np.dot(new_u.T,self.x) / np.sum(new_u.T, axis=1, keepdims=True)


 
    def _capnhat_mttv(self):
        """Cập nhật ma trận thành viên """
        kcach= np.zeros((self.n_data,self.n_clusters))
        
        # tính khoảng cách từ điểm dữ liệu đến các tâm cụm
        for i in range (self.n_clusters):
            kcach[:,i]= np.linalg.norm(self.x-self.centroids[i],axis=1)
        
        # cập nhật mức độ thành viên
        for i in range (self.n_data):
            for j in range(self.n_clusters):
               self.u[i,j] = 1.0 /np.sum((kcach[i,j]/ kcach[i,:])** (2/(self.m-1)))
        
    def _sai_so(self, old_u):
        """tính sự chênh lệch giữa ma trận thành viên cũ và ma trận thành viên mới """
        return np.linalg.norm(self.u - old_u)


    def fit(self):
        """thuật toán FCM """
        for i in range(self.max_iter):
            old_u = self.u.copy()
            self._capnhat_tamcum()  # Bước 2 : cập nhật tâm cụm
            self._capnhat_mttv()     # Bước 3: cập nhật ma trận thành viên
            if self._sai_so(old_u) < self.epsilon:  # Bước 4: kiểm tra điều kiện hội tụ
                break
        return self.u, self.centroids

#Dữ liệu đầu vào
x = [1, 3, 5, 7, 9]
n_clusters = 2

#Thực hiện thuật toán FCM
fcm = FCM(x, n_clusters)
u, centroids = fcm.fit()

print("Ma trận thành viên u:")
print(u)

print("Tâm cụm :")
print(centroids.flatten())




