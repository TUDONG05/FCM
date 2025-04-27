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
## • Yêu cầu hệ thống 
- Python 3.8 hoặc mới hơn
- Các thư viện Python cần thiết:
  - `numpy`
  - `matplotlib` (nếu cần vẽ đồ thị)
## • Clone repository: 
`git clone https://github.com/TUDONG05/BTL_python.git`

## • Cài đặt môi trường ảo:
`python -m venv env`
### Trên macOS/Linux
`source env/bin/activate`
### Trên Windows
`env\Scripts\activate`      
## • Cài đặt các thư viện cần thiết
` pip install -r requirements.txt1`
## • Chạy chương trình  
### -  Mở fcm_for.py hoặc fcm_np.py và sử dụng câu lệnh :
`python fcm_for.py` hoặc `python fcm_np.py`

## - Chất lượng phân cụm thu được 
### DATASET UCI id= 53 Iris 150 x 4
### Thời gian lấy dữ liệu: 0.004
### size=150 x 4
### Alg     Time    Step    DI+     DB-     PC+     XB-     CE-     SI+     FHV-
### FCM     0.0     14      0.66    0.67    0.674   0.11    0.396   0.549   0.783

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
