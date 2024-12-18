import numpy as np
# buoc 1:khoi tao ma tran thanh vien

class FCM:
    def __init__(self, x, n_clusters, m=2, max_iter=100, epsilon=1e-5):
        self.x = np.array(x, dtype=np.float64).reshape(-1, 1) #diem du lieu
        self.n_clusters = n_clusters  #socum
        self.m = m  #chi so mo
        self.max_iter = max_iter #lap toi da  
        self.epsilon = epsilon  #sai so epsilon
        self.n_data = self.x.shape[0] #so diem du lieu
        self.u = self._ktmttv() #ma tran thanh vien
        self.centroids = np.zeros((self.n_clusters,1))#tam cum

    def _ktmttv(self):
        np.random.seed(42)
        u =  np.random.rand(self.n_data,self.n_clusters)
        u=u / np.sum(u,axis=1, keepdims=True)
        #  u chia cho tong cac muc do phu thuoc thanh phan de chuan hoa sao cho tong cac phan tu cung 1 hang bang 1
        return u
    # buoc 2: cap nhat tam cum
    def _capnhat_tamcum(self):
        new_u=self.u**self.m
        self.centroids = np.dot(new_u.T,self.x) / np.sum(new_u.T, axis=1, keepdims=True)


    # buoc 3:cap nhat ma tran thanh vien
    def _capnhat_mttv(self):
        kcach= np.zeros((self.n_data,self.n_clusters))
        
        # tinh khoang cach tu diem dl den cac tam cum
        for i in range (self.n_clusters):
            kcach[:,i]= np.linalg.norm(self.x-self.centroids[i],axis=1)
        
        # cap nhat muc do thanh vien
        for i in range (self.n_data):
            for j in range(self.n_clusters):
               self.u[i,j] = 1.0 /np.sum((kcach[i,j]/ kcach[i,:])** (2/(self.m-1)))
        
    def _sai_so(self, old_u):
        return np.linalg.norm(self.u - old_u)
    def fit(self):
        for i in range(self.max_iter):
            old_u = self.u.copy()
            self._capnhat_tamcum()  # Cập nhật tâm cụm
            self._capnhat_mttv()  # Cập nhật ma trận thành viên
            if self._sai_so(old_u) < self.epsilon:  # Kiểm tra hội tụ
                break
        return self.u, self.centroids

# Dữ liệu đầu vào
x = [1, 3, 5, 7, 9]
n_clusters = 2

# Thực hiện FCM
fcm = FCM(x, n_clusters)
u, centroids = fcm.fit()

print("Ma trận thành viên (u):")
print(u)

print("Tâm cụm:")
print(centroids.flatten())




