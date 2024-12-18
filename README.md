# Ứng dụng lập trình hướng đối tượng và Numpy trong Python để triển khai các thuật toán phân cụm.
# 1.Giới thiệu:
## - Thuật toán FCM (Fuzzy C-Means) là một phương pháp phân cụm thuộc nhóm clustering trong Machine Learning. 
## - FCM hoạt động dựa trên nguyên lý rằng mỗi điểm dữ liệu có thể thuộc về nhiều cụm với các mức độ khác nhau, được mô tả qua ma trận thành viên. Cập nhật các cụm và ma trận thành viên được thực hiện qua các bước lặp cho đến khi hội tụ.

# Quy trình thực hiện FCM
## Bước 1: Khởi tạo
• Chọn số lượng cụm C( do người dùng chỉ định)
• Khởi tạo ngẫu nhiên các tâm cụm
• Khởi tạo ma trận thành viên U ngẫu nhiên, đảm bảo với tổng mức độ thành viên của một điểm dữ liệu trên tất cả các cụm bằng 1: 
         ![image](https://github.com/user-attachments/assets/44ea30bf-a58f-4592-84b0-f31603f55e04)



# Bước 2:Cập nhật tâm cụm
![image](https://github.com/user-attachments/assets/89f09dae-dfa2-45d2-acfd-49c50d745335)

## Trong đó: 
• vj : là tâm cụm thứ j
• uij: mức độ thành viên của điểm dữ liệu i đối với cụm j.
• N:là số điểm dữ liệu
• xi: điểm dữ liệu thứ i.
• m:chỉ số  mờ.
# Bước 3 Cập nhật ma trận thành viên
- Tính mức độ thành viên  cho từng điểm dữ liệu bằng công thức
![image](https://github.com/user-attachments/assets/c6abfc93-4f8d-4121-a4c8-ae5ce436ac5b)

Trong đó: 
• c là số cụm
• xi: điểm dữ liệu thứ I
• vj: tâm cụm thứ j
• m: chỉ số mờ



# Bước 4:Kiểm tra điều kiện hội tụ:
Nếu ma trận thành viên U thay đổi không đáng kể (theo ngưỡng epsilon) hoặc số vòng lặp đạt mức tối đa, thì dừng thuật toán.
