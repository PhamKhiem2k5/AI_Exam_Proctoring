"""
Hệ thống phát hiện hướng đầu (Head Pose) nâng cao – có bộ đếm thời gian và ghi log
Tính năng mới:
- Đếm thời gian vi phạm liên tục (quay đầu, cúi, ngửa, mất mặt)
- Chỉ cảnh báo khi vi phạm kéo dài > ngưỡng (mặc định 2 giây)
- Ghi log vi phạm vào file CSV
- Hiển thị trạng thái chi tiết trên màn hình
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime
import time

# ---------------------- CẤU HÌNH ----------------------
VIOLATION_THRESHOLD = 2.0      # Số giây vi phạm liên tục trước khi cảnh báo
NO_FACE_THRESHOLD = 3.0        # Số giây mất mặt trước khi cảnh báo
YAW_THRESHOLD = 25             # Quay hẳn ra sau (độ)
YAW_SIDE_THRESHOLD = 10        # Quay sang trái/phải (độ)
PITCH_THRESHOLD = 10           # Cúi hoặc ngửa (độ)

# ---------------------- KHỞI TẠO MEDIAPIPE ----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------- KHỞI TẠO CAMERA ----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam. Kiểm tra kết nối.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------------- BIẾN LƯU TRẠNG THÁI ----------------------
violation_start_time = None       # Thời điểm bắt đầu vi phạm hiện tại
current_violation_type = None     # Loại vi phạm đang diễn ra
last_warning_time = 0             # Thời gian cảnh báo cuối cùng (để tránh spam)
warning_cooldown = 1.0            # Chỉ cảnh báo 1 lần/giây

# Lịch sử vi phạm (log)
violation_log = []
log_filename = f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Mở file log và ghi header
with open(log_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Thời gian', 'Loại vi phạm', 'Thời gian kéo dài (giây)', 'Ảnh chụp'])

# ---------------------- HÀM GHI LOG ----------------------
def log_violation(violation_type, duration, frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Chụp ảnh màn hình
    img_path = f"screenshots/{timestamp.replace(':', '-')}.jpg"
    cv2.imwrite(img_path, frame)
    with open(log_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, violation_type, f"{duration:.2f}", img_path])
    print(f"📝 Log: {timestamp} - {violation_type} ({duration:.2f}s)")

# ---------------------- VÒNG LẶP CHÍNH ----------------------
print("🎥 Đang mở webcam. Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Biến trạng thái trong khung hình hiện tại
    warning_text = "✅ Bình thường"
    color = (0, 255, 0)
    angle_text = ""
    detected_violation = None   # Loại vi phạm nếu có

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        key_indices = [1, 199, 33, 263, 61, 291]
        face_2d = []
        face_3d = []

        for idx in key_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            z = lm.z
            face_2d.append([x, y])
            face_3d.append([x, y, z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        camera_matrix = np.array([
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, camera_matrix, dist_coeffs
        )
        if success:
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

            pitch = angles[0] * 360
            yaw = angles[1] * 360
            angle_text = f"Pitch: {int(pitch)} | Yaw: {int(yaw)}"

            # Xác định loại vi phạm (ưu tiên mức độ nghiêm trọng)
            if yaw > YAW_THRESHOLD or yaw < -YAW_THRESHOLD:
                detected_violation = "Quay hẳn ra sau"
            elif yaw > YAW_SIDE_THRESHOLD:
                detected_violation = "Quay sang trái"
            elif yaw < -YAW_SIDE_THRESHOLD:
                detected_violation = "Quay sang phải"
            elif pitch > PITCH_THRESHOLD:
                detected_violation = "Ngửa mặt lên"
            elif pitch < -PITCH_THRESHOLD:
                detected_violation = "Cúi xuống"
            else:
                detected_violation = None
        else:
            # Nếu không giải được PnP (hiếm khi xảy ra)
            detected_violation = None
    else:
        # Không thấy mặt
        detected_violation = "Mất mặt / Che camera"

    # ------------------ XỬ LÝ THỜI GIAN VI PHẠM ------------------
    current_time = time.time()

    if detected_violation:
        # Nếu đang có vi phạm
        if violation_start_time is None:
            # Bắt đầu vi phạm mới
            violation_start_time = current_time
            current_violation_type = detected_violation
        elif detected_violation != current_violation_type:
            # Thay đổi loại vi phạm -> reset đếm
            violation_start_time = current_time
            current_violation_type = detected_violation

        # Tính thời gian vi phạm
        duration = current_time - violation_start_time

        # Nếu vượt ngưỡng và chưa cảnh báo quá gần đây
        if (duration > VIOLATION_THRESHOLD and 
            current_time - last_warning_time > warning_cooldown):
            # Ghi log và cảnh báo
            log_violation(current_violation_type, duration, frame)
            last_warning_time = current_time
            # Cập nhật màu đỏ
            color = (0, 0, 255)
            warning_text = f"⚠️ {current_violation_type} ({duration:.1f}s)"

        else:
            # Đang vi phạm nhưng chưa đủ thời gian hoặc đã cảnh báo gần đây
            color = (0, 0, 255) if duration > VIOLATION_THRESHOLD else (0, 255, 255)
            warning_text = f"🟡 {current_violation_type} ({duration:.1f}s)" if duration > 0 else "🔴 Đang vi phạm"

    else:
        # Không vi phạm, reset các biến
        if violation_start_time is not None:
            # Nếu trước đó đang vi phạm, reset
            violation_start_time = None
            current_violation_type = None
        warning_text = "✅ Bình thường"
        color = (0, 255, 0)

    # ------------------ HIỂN THỊ TRÊN MÀN HÌNH ------------------
    cv2.putText(frame, warning_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if angle_text:
        cv2.putText(frame, angle_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2, cv2.LINE_AA)

    # Hiển thị thời gian vi phạm (nếu có)
    if violation_start_time is not None and current_violation_type:
        duration = current_time - violation_start_time
        cv2.putText(frame, f"Thời gian: {duration:.1f}s", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Exam Proctoring - Advanced", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ---------------------- GIẢI PHÓNG ----------------------
cap.release()
cv2.destroyAllWindows()
print(f"📊 Log đã được lưu tại: {log_filename}")
print("🔴 Đã đóng chương trình.")