from statistics import mode
import cv2
import os
from keras.models import load_model
import numpy as np
from ultralytics import YOLO

# Đường dẫn model
yolo_model_path = './trained_models/detection_models/yolov12n-face.pt'
emotion_model_path = './trained_models/emotion_models/fine_tuned_miniXCEPTION_final.hdf5'

# Load YOLO
print("🔄 Đang load YOLO face detection model...")
yolo_model = YOLO(yolo_model_path)

# Load model emotion
print("🔄 Đang load emotion model...")
if not os.path.exists(emotion_model_path):
    print(f"❌ File model không tồn tại: {emotion_model_path}")
    exit()

emotion_classifier = load_model(emotion_model_path, compile=False, safe_mode=False)
print("✅ Emotion model loaded thành công")

# Lấy input size và số lớp
emotion_target_size = emotion_classifier.input_shape[1:3]

# Tạo dictionary nhãn theo thứ tự bạn cung cấp
emotion_labels = {
    0: 'angry',
    1: 'bored',
    2: 'disgust',
    3: 'drowsy',
    4: 'fear',
    5: 'happy',
    6: 'looking_away',
    7: 'neutral',
    8: 'sad',
    9: 'suprise'
}

# Hàm sinh màu khác nhau cho từng nhãn
def get_color(label_index):
    np.random.seed(label_index)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))

# Hyper-parameters
frame_window = 10
emotion_window = []

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
        break

    results = yolo_model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            face = frame[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            try:
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except:
                continue

            gray_face = gray_face.astype('float32') / 255.0
            gray_face = (gray_face - 0.5) * 2.0
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            prediction = emotion_classifier.predict(gray_face, verbose=0)
            emotion_label_arg = np.argmax(prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            emotion_window.append(emotion_text)
            if len(emotion_window) > frame_window:
                emotion_window.pop(0)

            try:
                emotion_mode = mode(emotion_window)
            except:
                emotion_mode = emotion_text

            color_bgr = get_color(list(emotion_labels.keys())[list(emotion_labels.values()).index(emotion_mode)])

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            # Vẽ text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(emotion_mode, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 10
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                          (text_x + text_size[0] + 5, text_y + 5), color_bgr, -1)
            cv2.putText(frame, emotion_mode, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    cv2.imshow('window_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
