import numpy as np
import matplotlib.pyplot as plt
# buoc 1:khoi tao ma tran thanh vien
def ktmttv(data,cluster):
    np.random.seed(42)
    u =  np.random.rand(data,cluster)
    u=u / np.sum(u,axis=1, keepdims=True)
    #  u chia cho tong cac muc do phu thuoc thanh phan de chuan hoa sao cho tong cac phan tu cung 1 hang bang 1
    return u
# buoc 2: cap nhat tam cum
def capnhat_tamcum(u,x,m):
    new_u=u**m
    centroid = np.dot(new_u.T,x) / np.sum(new_u.T, axis=1, keepdims=True)
    return centroid

# buoc 3:cap nhat ma tran thanh vien
def capnhat_mttv(u,x,centroid,m):
    data,cluster = u.shape
    kcach= np.zeros((data,cluster))
    
    # tinh khoang cach tu diem dl den cac tam cum
    for i in range (cluster):
        kcach[:,i]= np.linalg.norm(x-centroid[i],axis=1)
    
    # cap nhat muc do thanh vien
    for i in range (data):
        for j in range(cluster):
            u[i,j] = 1.0 /np.sum((kcach[i,j]/ kcach[i,:])** (2/(m-1)))
    return u

def fcm(x,cluster,m=2,max_iter =100,epsilon=1e-5):
    data = x.shape[0]
    x = x.reshape(-1, 1)
    u = ktmttv(data,cluster)

    for i in range (max_iter):
        old_u = u.copy()
        centroid =capnhat_tamcum(u,x,m)
        u = capnhat_mttv(u,x,centroid,m)

        # buoc 4: kiem tra dieu kien hoi tu
        if np.linalg.norm(u-old_u) < epsilon:
            break
    return u,centroid

x = np.array([1,3,5,7,9])
# 5 diem du lieu 1,3,5,7,9
cluster = 2
# so cum: 2

#  thuc hien fcm
u,centroid = fcm(x,cluster,m=2)
# Hien thi ket qua
print("Ma tran thanh vien (u):")
print(u)

print("Trung tam cum:")
print(centroid)






