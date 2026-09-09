import cv2

# Khởi tạo kết nối với webcam (số 0 thường là camera mặc định của máy)
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có mở thành công không
if not cap.isOpened():
    print("Lỗi: Không thể kết nối với Camera!")
    exit()

print("Camera đã mở thành công! Nhấn phím 'q' trên cửa sổ video để thoát.")

# Vòng lặp để đọc và hiển thị video liên tục
while True:
    # Đọc từng khung hình (frame) từ camera
    ret, frame = cap.read()
    
    # Nếu không đọc được khung hình, thoát vòng lặp
    if not ret:
        print("Lỗi: Không thể nhận hình ảnh từ Camera.")
        break

    # Hiển thị khung hình lên cửa sổ có tên "Test Camera"
    cv2.imshow("Test Camera", frame)

    # Đợi 1 millisecond, nếu người dùng nhấn phím 'q' thì thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Đang đóng Camera...")
        break

# Giải phóng tài nguyên camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()