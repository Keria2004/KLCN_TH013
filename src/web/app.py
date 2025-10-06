from flask import Flask, render_template, request, send_file, Response
import cv2
import os
import numpy as np
from statistics import mode
from keras.models import load_model
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------- Load models -------------------
yolo_model_path = './trained_models/detection_models/yolov12n-face.pt'
emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

emotion_labels = get_labels('fer2013')
frame_window = 10

yolo_model = YOLO(yolo_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False, safe_mode=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# ------------------- Helper -------------------
def process_frame(frame):
    emotion_window = []
    results = yolo_model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            gray_face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            try:
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except:
                continue
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            pred = emotion_classifier.predict(gray_face, verbose=0)
            emotion_probability = np.max(pred)
            emotion_text = emotion_labels[np.argmax(pred)]
            emotion_window.append(emotion_text)
            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                emotion_mode = emotion_text

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, emotion_mode, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame

# ------------------- Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return "No file uploaded", 400

    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(filepath)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        frame = cv2.imread(filepath)
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    elif ext in ['.mp4', '.avi', '.mov']:
        def generate():
            cap = cv2.VideoCapture(filepath)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            cap.release()
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Unsupported file type", 400

if __name__ == '__main__':
    app.run(debug=True)