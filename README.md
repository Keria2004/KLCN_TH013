# 🎭 Face Classification (Emotion & Gender)

Dự án nhận diện **khuôn mặt + cảm xúc + giới tính** sử dụng **Keras CNN** và **OpenCV**.  
Model huấn luyện trên datasets:

- **FER2013** → Emotion classification (Accuracy ~66%)
- **IMDB** → Gender classification (Accuracy ~96%)

---

## 🚀 Tính năng

- Nhận diện khuôn mặt realtime từ webcam
- Phân loại cảm xúc (angry, disgust, fear, happy, sad, surprise, neutral)
- Demo ảnh, video
- Hỗ trợ **fine-tuning** trên dữ liệu mới

# Cài thư viện

pip install -r requirements.txt

# Realtime Emotion Detection (Webcam)

python src/video_emotion_color_demo.py

# Train lại

python src/train_emotion_classifier.py

# Tài liệu tham khảo

https://github.com/oarriaga/face_classification
https://github.com/YapaLab/yolo-face/tree/dev
