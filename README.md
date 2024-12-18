# Ứng dụng lập trình hướng đối tượng và Numpy trong Python để triển khai các thuật toán phân cụm.
# 1.Giới thiệu:
## - Thuật toán FCM (Fuzzy C-Means) là một phương pháp phân cụm thuộc nhóm clustering trong Machine Learning. 
## - FCM hoạt động dựa trên nguyên lý rằng mỗi điểm dữ liệu có thể thuộc về nhiều cụm với các mức độ khác nhau, được mô tả qua ma trận thành viên. Cập nhật các cụm và ma trận thành viên được thực hiện qua các bước lặp cho đến khi hội tụ.

## 2.Các bước chính của thuật toán:
1. **Khởi tạo ma trận thành viên**: Bắt đầu với một ma trận ngẫu nhiên và chuẩn hóa sao cho mỗi hàng của ma trận có tổng bằng 1.
2. **Cập nhật tâm cụm**: Dựa trên ma trận thành viên và các điểm dữ liệu, tính toán lại vị trí các tâm cụm.
3. **Cập nhật ma trận thành viên**: Dựa trên khoảng cách giữa các điểm dữ liệu và tâm cụm, tính toán lại mức độ thuộc về mỗi cụm của các điểm dữ liệu.
4. **Kiểm tra sai số**: Nếu sự thay đổi giữa các ma trận thành viên giữa các vòng lặp nhỏ hơn một giá trị epsilon, thuật toán sẽ hội tụ và dừng lại.
