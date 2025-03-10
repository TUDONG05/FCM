import numpy as np
import time 
from scipy.spatial.distance import cdist
from utility import *
class FCM:
    def __init__(self, X, n_clusters, m, max_iter, epsilon,seed):
        # self.x = np.array(x, dtype=np.float64)# các điểm dữ liệu
        self.X = np.array(X)  # ma tran diem du lieu [n_data x n_features]
        self.n_clusters = n_clusters  # số cụm
        self.m = m  # chỉ số mờ
        self.seed = seed
        self.max_iter = max_iter # số lần lặp tối  đa
        self.epsilon = epsilon  # sai số epsilon
        self.n_data ,self.n_features = self.X.shape
        self.u = self._ktmttv() #ma trận thành viên
        
        self.centroids = self._khoitao_tamcum()
        self.time = 0
        self.step = 0
        

    def _khoitao_tamcum(self):
        """Chọn ngẫu nhiên n_clusters điểm từ dữ liệu làm tâm cụm ban đầu"""
        np.random.seed(self.seed)
        return self.X[np.random.choice(self.n_data, self.n_clusters, replace=False)]


    def _ktmttv(self):
        """Khởi tạo ma trận thành viên """
        np.random.seed(self.seed)
        u =  np.random.rand(self.n_data,self.n_clusters)
        u=u / np.sum(u,axis=1, keepdims=True)
        # u chia cho tổng các mức độ phụ thuộc thành phần để chuẩn hóa đảm bảo tổng các phần từ cùng 1 hàng bằng 1
        return u
    
    
    def _capnhat_tamcum(self):
        """cập nhật tâm cụm """
        new_u=self.u**self.m
        return np.round( np.dot(new_u.T,self.X) / np.sum(new_u.T, axis=1, keepdims=True),2)

    
 
    # def _capnhat_mttv(self):
    #     """Cập nhật ma trận thành viên """
    #     kcach= np.zeros((self.n_data,self.n_clusters))
        
    #     # tính khoảng cách từ điểm dữ liệu đến các tâm cụm
    #     for i in range (self.n_clusters):
    #         kcach[:,i]= np.linalg.norm(self.X-self.centroids[i],axis=1)
        
    #     # cập nhật mức độ thành viên
    #     for i in range (self.n_data):
    #         for j in range(self.n_clusters):
    #            kcach[i, j] = np.where(kcach[i, j] == 0, np.finfo(float).eps, kcach[i, j])
    #            self.u[i,j] = 1.0 /np.sum((kcach[i,j]/ kcach[i,:])** (2/(self.m-1)))
    #     return self.u




    def _capnhat_mttv(self):
        """Cập nhật ma trận thành viên U theo công thức FCM"""
        d = division_by_zero(distance_cdist(self.X,self.centroids) ) #n,c
        d_exp = (d[:, :, None] / d[:, None, :]) ** (2 / (self.m - 1))  # shape = (n_data, n_clusters, n_clusters)

        sum_d = np.sum(d_exp, axis=2)

        return 1 / sum_d  # shape = (n_data, n_clusters)





    def _sai_so(self, old_u):
        """tính sự chênh lệch giữa ma trận thành viên cũ và ma trận thành viên mới """
        return np.linalg.norm(self.u - old_u)


    def fit(self):
        """thuật toán FCM """
        time_start = time.time()
        for i in range(self.max_iter):
            self.step+=1
            old_u = self.u.copy()
            self.centroids = self._capnhat_tamcum()  # Bước 2 : cập nhật tâm cụm
            self.u= self._capnhat_mttv()     # Bước 3: cập nhật ma trận thành viên
            if self._sai_so(old_u) < self.epsilon:  # Bước 4: kiểm tra điều kiện hội tụ
                break
        time_end = time.time()
        self.time = time_end - time_start
        return self.u, self.centroids, self.step
        
    
if __name__ == '__main__':
    import time
    from utility import round_float, extract_labels
    from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
    from validity import dunn, davies_bouldin, partition_coefficient,Xie_Benie,classification_entropy, silhouette,hypervolume,accuracy_score

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 100
    M = 2
    SEED = 42
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report_ssfcm(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            alg,
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(Xie_Benie(X, V, U)),  # XB
            wdvl(classification_entropy(U)), # CE
            wdvl(silhouette(X, labels)), #SI
            wdvl(hypervolume(U,m=2)), # FHV
        
        ]
        return SPLIT.join(kqdg)
    # =======================================

    clustering_report = []
    data_id = 53
    if data_id in TEST_CASES:
        _start_time = time.time()
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        _size = f"{_dt['data']['num_instances']} x {_dt['data']['num_features']}"
        print(f'size={_size}')
        n_clusters = _TEST['n_cluster']
        # ===============================================

        
        fcm = FCM(X, n_clusters=n_clusters, m=M, max_iter=MAX_ITER, epsilon=EPSILON,seed=SEED)
        u,centroids,step=fcm.fit()

        # print('Ma tran u ngang: ', u_ngang)
        # print('Ma tran nhan: ', labels_all),
        print('Ma tran tam cum: ', centroids)
        # print ('Ma tran thanh vien: ',u)

        
        # --------------------------------  
        titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'XB-', 'CE-','SI+','FHV-']
        print(SPLIT.join(titles))
        print(write_report_ssfcm(alg='FCM', index=0, process_time=fcm.time, step=step, X=X, V=fcm.centroids, U=fcm.u))


        