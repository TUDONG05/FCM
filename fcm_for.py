import random
import numpy as np
class FCM:
    def __init__(self, data, n_clusters, m=2, max_iter=100, epsilon=1e-5):
        self.data = data  #diem du lieu
        self.n_clusters = n_clusters  #socum
        self.m = m  #chi so mo
        self.max_iter = max_iter #lap toi da  
        self.epsilon = epsilon  #sai so epsilon
        self.n_data = len(data)  #so diem du lieu
        self.u = self.ktmttv() #ma tran thanh vien
        self.centroids = [0] * self.n_clusters  #tam cum

        
    
    def ktmttv(self):
        """khoi tao ma tran thanh vien"""
        random.seed(42)
        u=[]
        for i in range(self.n_data):
            hang=[]
            for j in range(self.n_clusters):
  
                hang.append(random.random())

# ta khoi tao cac gia tri ngau nhien tren tung hang,
# chuan hoa sao cho cac tong cac ptu trong 1 hang = 1     
#  sau do them cac hang do vao list u
           
            tong_hang=sum(hang)
            for k in range(len(hang)):
                hang[k]/=tong_hang

            u.append(hang)
        return u

    def capnhat_tamcum(self):
        """cap nhat tam cum"""
        for j in range (self.n_clusters):
            tu =0
            mau =0
            for i in range (self.n_data):
                u_m=self.u[i][j]**self.m
                tu +=u_m *self.data[i]
                mau+=u_m
            self.centroids[j]=tu/mau if mau !=0 else 0



    def khoang_cach(self):
        """tinh khoang cach tu diem dl den cac tam cum"""
        kcach=[]
        for i in range (self.n_data):
            kc_hang=[]
            for j in range (self.n_clusters):
                kc_hang.append(abs(self.data[i]-self.centroids[j]))
            kcach.append(kc_hang)
        return kcach

    def capnhat_mttv(self):
        """cap nhat ma tran thanh vien"""

        kc = self.khoang_cach()
        for i in range(self.n_data):
            for j in range(self.n_clusters):
                    M=0
                    for k in range(self.n_clusters):
                        t =kc[i][j]
                        m =kc[i][k]

                        M+=(t/m) **2/(self.m-1)
                    self.u[i][j] =1/M
    def sai_so(self, old_u):
        """sai so giua cu va moi  de kiem tra dieu kien hoi tu"""
        ss = 0
        for i in range(self.n_data):
            for j in range(self.n_clusters):
                ss += abs(self.u[i][j] - old_u[i][j])
        return ss
    

    def fcm(self):
        for _ in range(self.max_iter):
            old_u=[]
            for hang in self.u:
                old_u.append(hang[:])

            self.capnhat_tamcum() # buoc 2: cap nhat tam cum
            self.capnhat_mttv()  # buoc 3:cap nhat ma tran thanh vien
            if self.sai_so(old_u) < self.epsilon:  # buoc  4: kiem tra dieu kien hoi tu
                break
        return self.u, self.centroids


# du lieu dau vao
data = [1, 3, 5, 7, 9]  
n_clusters = 2  

# thuc hien fcm 
fcm = FCM(data, n_clusters)
u, centroids = fcm.fcm()


print("Ma tran thanh vien (u):")
for data in u:
    print(data)

print("Tam cum:")
print(centroids)

        