import cv2
from ultralytics import YOLO
import mediapipe as mp

# --- 1. Khởi tạo model ---
model = YOLO('models/best.pt')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# --- 2. Mở camera ---
cap = cv2.VideoCapture(0)

# --- 3. Các tham số tối ưu ---
FRAME_SKIP = 3  # Xử lý YOLO mỗi 3 frame
CONFIDENCE_THRESHOLD = 0.5
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- 4. MediaPipe chạy liên tục (phân tích hành vi) ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        # TODO: Thêm logic phân tích tư thế, ánh mắt
        pass

    # --- 5. YOLO chạy ngắt quãng (phát hiện vật thể) ---
    if frame_count % FRAME_SKIP == 0:
        # Dự đoán trên frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Vẽ kết quả lên frame
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    # --- 6. Hiển thị ---
    cv2.imshow('AI Exam Proctoring System', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()