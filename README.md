# Ứng dụng lập trình hướng đối tượng và Numpy trong Python để triển khai các thuật toán phân cụm mờ .
# 1.Giới thiệu:
![img_22.png](img_22.png)


## - Thuật toán FCM (Fuzzy C-Means) là một phương pháp phân cụm thuộc nhóm clustering trong Machine Learning. 
## - FCM hoạt động dựa trên nguyên lý rằng mỗi điểm dữ liệu có thể thuộc về nhiều cụm với các mức độ khác nhau, được mô tả qua ma trận thành viên. Cập nhật các cụm và ma trận thành viên được thực hiện qua các bước lặp cho đến khi hội tụ.

# 2.Quy trình thực hiện FCM
## Bước 1: Khởi tạo
• Chọn số lượng cụm C( do người dùng chỉ định)
• Khởi tạo ngẫu nhiên các tâm cụm
• Khởi tạo ma trận thành viên u ngẫu nhiên, đảm bảo với tổng mức độ thành viên của một điểm dữ liệu trên tất cả các cụm bằng 1: 

![image](https://github.com/user-attachments/assets/965f90a6-f3b7-4587-a896-7de84dbaed2c)

## Bước 2:Cập nhật tâm cụm
![image](https://github.com/user-attachments/assets/89f09dae-dfa2-45d2-acfd-49c50d745335)

## Trong đó: 
### • vj : là tâm cụm thứ j
### • uij: mức độ thành viên của điểm dữ liệu i đối với cụm j.
### • N:là số điểm dữ liệu
### • xi: điểm dữ liệu thứ i.
### • m:chỉ số  mờ.

## Bước 3 Cập nhật ma trận thành viên
## - Tính mức độ thành viên  cho từng điểm dữ liệu bằng công thức
![image](https://github.com/user-attachments/assets/735d2acc-9aa1-456c-ac47-0b2c222b1f29)


## Trong đó: 
### • c là số cụm
### • xi: điểm dữ liệu thứ I
### • vj: tâm cụm thứ j
### • m: chỉ số mờ

## Bước 4:Kiểm tra điều kiện hội tụ:
### • Nếu ma trận thành viên U thay đổi không đáng kể (theo ngưỡng epsilon) hoặc số vòng lặp đạt mức tối đa, thì dừng thuật toán.
# 3. Cài đặt thuật toán bằng ngôn ngữ Python 
## a) Sử dụng thư viện numpy kết hợp với lập trình hướng đối tượng 
### • Nhập thư viện numpy 
![img_3.png](img_3.png)
### • Khai báo lớp FCM và hàm khởi tạo: 
![img_2.png](img_2.png) 
### • Hàm khởi tạo ma trận thành viên ngẫu nhiên 
![img_4.png](img_4.png)
### • Hàm cập nhật tâm cụm 
![img_5.png](img_5.png)
### • Hàm cập nhật ma trận thành viên 
![img_6.png](img_6.png)
### • Hàm tính sự chênh lệch giữa ma trận thành viên cũ và ma trận thành viên mới 
![img_7.png](img_7.png)
### • Hàm triển khai thuật toán Fuzzy C-Means 
![img_8.png](img_8.png)
### • Truyền dữ liệu và thực hiện 
![img_9.png](img_9.png)
### • Kết quả thu được 
![img_10.png](img_10.png)

## b) Sử dụng vòng lặp for kết hợp với lập trình hướng đối tượng 
### • Nhập thư viện random 
![img_11.png](img_11.png)
### • Khai báo lớp FCM và hàm khởi tạo
![img_13.png](img_13.png)
### • Hàm khởi tạo ma trận thành viên ngẫu nhiên 
![img_14.png](img_14.png)
### • Hàm cập nhật tâm cụm 
![img_15.png](img_15.png)
### • Hàm tính khoảng cách từ điểm dữ liệu đến tâm cụm 
![img_23.png](img_23.png)
### • Hàm cập nhật ma trận thành viên 
![img_17.png](img_17.png)
### • Hàm tính sự chênh lệch giữa ma trận thành viên cũ và ma trận thành viên mới 
![img_18.png](img_18.png)
### • Hàm triển khai thuật toán Fuzzy C-Means 
![img_24.png](img_24.png)
### • Truyền dữ liệu và thực hiện 
![img_20.png](img_20.png)
### • Kết quả thu được 
![img_21.png](img_21.png)
# 4. Ưu điểm của FCM:
## • Xử lý mờ: FCM phù hợp với dữ liệu không có ranh giới rõ ràng, giúp phân cụm linh hoạt hơn K-Means.
## • Thông tin bổ sung: Mức độ thành viên cung cấp thêm thông tin về mối quan hệ giữa điểm dữ liệu và các cụm.
# 5. Hạn chế của FCM
   ## • Phụ thuộc vào m: Chỉ số mờ m cần được chọn phù hợp, vì nó ảnh hưởng đến kết quả.
   ## • Nhạy cảm với điểm nhiễu: Giống K-Means, FCM dễ bị ảnh hưởng bởi các điểm dữ liệu nằm xa (outliers).
   ## • Số cụm cố định: Người dùng cần chỉ định trước số cụm C, có thể không phù hợp với dữ liệu thực tế.
# 6. Ứng dụng
## Xử lý ảnh: 
### • Phân đoạn ảnh y tế (ví dụ: tách vùng não hoặc tế bào ung thư)
### •  Nhận dạng mẫu:phân loại các đối tượng trong ảnh(ví dụ người, động vật, đồ vật, ...)

## Khai phá dữ liệu:
### • Phân tích thị trường: Phân nhóm khách hàng dựa trên hành vi mua sắm, sở thích. 
### • Phân tích văn bản: Phân loại văn bản thành các chủ đề khác nhau. 
### • Phân tích gen: Phân nhóm các gen dựa trên biểu hiện gen.
