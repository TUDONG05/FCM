import random
class FCM:
    def __init__(self, data, n_clusters, m=2, max_iter=100, epsilon=1e-5):
        self.data = data  #điểm dữ liệu
        self.n_clusters = n_clusters  #số cụm
        self.m = m  #chỉ số mờ
        self.max_iter = max_iter #số lần lặp tối đa
        self.epsilon = epsilon  # sai số epsilon
        self.n_data = len(data)  #số điểm dữ liệu
        self.u = self.ktmttv() #ma trận thành viên
        self.centroids = [0] * self.n_clusters  #tâm cụm

    def ktmttv(self):
        """Khởi tạo ma trận thành viên """
        random.seed(42)
        u=[]
        for i in range(self.n_data):
            hang=[]
            for j in range(self.n_clusters):
                hang.append(random.random()) # ta khởi tạo các giá trị ngẫu nhiên trên từng hàng
            tong_hang=sum(hang)
            for k in range(len(hang)):
                hang[k]/=tong_hang # chuẩn hóa đảm bảo tổng các phần tử trong hàng = 1
            u.append(hang) # Sau đó thêm từ hàng vào ma trận thành viên
        return u

    def capnhat_tamcum(self):
        """cập nhât tâm cụm """
        for j in range (self.n_clusters):
            tu =0
            mau =0
            for i in range (self.n_data):
                u_m=self.u[i][j]**self.m
                tu +=u_m *self.data[i]
                mau+=u_m
            self.centroids[j]=tu/mau if mau !=0 else 0



    def khoang_cach(self):
        """tính khoảng các từ điểm dữ liệu đến tâm cụm """
        kcach=[]
        for i in range (self.n_data):
            kc_hang=[]
            for j in range (self.n_clusters):
                kc_hang.append(abs(self.data[i]-self.centroids[j]))
            kcach.append(kc_hang)
        return kcach

    def capnhat_mttv(self):
        """cập nhật ma trận thành viên """

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
        """tính sự chênh lệch giữa ma trận thành viên cũ và ma trận thành viên mới """
        ss = 0
        for i in range(self.n_data):
            for j in range(self.n_clusters):
                ss += abs(self.u[i][j] - old_u[i][j])
        return ss
    

    def fit (self):
        """thuật toán fcm """
        for _ in range(self.max_iter):
            old_u=[]
            for hang in self.u:
                old_u.append(hang[:])

            self.capnhat_tamcum() # bước 2 : cập nhật tâm cụm
            self.capnhat_mttv()  # bước 3 : cập nhật ma trận thành viên
            if self.sai_so(old_u) < self.epsilon:  # bước 4 : kiểm tra điều kiện hội tụ
                break
        return self.u, self.centroids


#dữ liệu đầu vào
data = [1, 3, 5, 7, 9]  
n_clusters = 2

#thực hiện thuật toán FCM
fcm = FCM(data, n_clusters)
u, centroids = fcm.fit()


print("Ma trận thành viên u :")
for data in u:
    print(data)

print("Tâm cụm :")
print(centroids)

        