from statistics import mode
import cv2
import os  # Để check file
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.preprocessor import preprocess_input
from ultralytics import YOLO

# Đường dẫn model (tương đối)
yolo_model_path = './trained_models/detection_models/yolov12n-face.pt'
emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

# Load nhãn emotion
emotion_labels = get_labels('fer2013')

# Hyper-parameters
frame_window = 10
emotion_offsets = (20, 40)  # Giữ nếu cần, nhưng không dùng ở đây
emotion_window = []

# Load model
print("🔄 Đang load YOLO face detection model...")
yolo_model = YOLO(yolo_model_path)

print("🔄 Đang load emotion model...")
# Kiểm tra file tồn tại
if not os.path.exists(emotion_model_path):
    print(f"❌ File model không tồn tại: {emotion_model_path}")
    exit()

try:
    emotion_classifier = load_model(emotion_model_path, compile=False, safe_mode=False, custom_objects={})
    print("✅ Emotion model loaded thành công")
except Exception as e:
    print(f"❌ Lỗi khi load model: {str(e)}")
    exit()

# Lấy input size của model
emotion_target_size = emotion_classifier.input_shape[1:3]

# Mở webcam
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ Không mở được webcam")
    exit()

print("✅ Webcam đã mở. Bấm 'q' để thoát.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Không lấy được frame từ camera")
        break

    # Chạy YOLO trên frame BGR gốc
    results = yolo_model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Debug: In tọa độ (xóa sau test)
            print(f"Box coords: ({x1}, {y1}, {x2}, {y2}) - Width: {x2-x1}, Height: {y2-y1}")

            # Cắt khuôn mặt
            gray_face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            try:
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            # Dự đoán cảm xúc
            emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            emotion_window.append(emotion_text)
            if len(emotion_window) > frame_window:
                emotion_window.pop(0)

            try:
                emotion_mode = mode(emotion_window)
            except:
                emotion_mode = emotion_text

            # Màu sắc theo emotion_mode (BGR)
            prob = emotion_probability
            if emotion_mode == 'angry':
                color_bgr = (0, 0, int(255 * prob))
            elif emotion_mode == 'sad':
                color_bgr = (int(255 * prob), 0, 0)
            elif emotion_mode == 'happy':
                color_bgr = (0, int(255 * prob), int(255 * prob))
            elif emotion_mode == 'surprise':
                color_bgr = (int(255 * prob), int(255 * prob), 0)
            elif emotion_mode == 'fear':
                color_bgr = (int(128 * prob), 0, int(128 * prob))
            elif emotion_mode == 'disgust':
                color_bgr = (0, int(128 * prob), 0)
            else:  # neutral
                color_bgr = (int(128 * prob), int(128 * prob), int(128 * prob))

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            # Vẽ text
            text = emotion_mode
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 10

            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                          (text_x + text_size[0] + 5, text_y + 5), color_bgr, -1)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    cv2.imshow('window_frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()