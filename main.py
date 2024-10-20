from flask import request, jsonify, render_template
from config import app
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import keras
import mtcnn
from mtcnn.mtcnn import MTCNN
from flask import send_from_directory


# Constants
MODEL_PATH = './model/masknet.h5'
CASCADE_PATH = './model/haarcascade_frontalface_default.xml'
MASK_LABEL = {0: 'MASK', 1: 'NO MASK'}
PROCESSED_FOLDER = 'static/processed/'
UPLOAD_FOLDER = 'static/upload/'

# Helper functions
def save_file(file, folder, prefix=''):
    filename = secure_filename(file.filename)
    filepath = os.path.join(folder, f"{prefix}_{filename}")
    file.save(filepath)
    return filepath


def load_model():
    print("Keras version:", keras.__version__)
    return keras.models.load_model(MODEL_PATH)



def process_image(img, model):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(img_rgb)
    mask_label = {0: 'MASK', 1: 'NO MASK'}

    for face in faces:
        x, y, w, h = face['box']
        crop = img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        label = mask_label[np.argmax(mask_result)]

        color = (255, 0, 0) if label == 'MASK' else (0, 0, 255)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return img


@app.route('/', methods=['GET'])
def mainpage():
    return render_template('base.html')

@app.route('/test_api', methods=['GET'])
def get():
    return jsonify({'msg': 'Hello, World!'})

@app.route('/mask_detection', methods=['POST'])
def mask_detection():
    if 'img' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['img']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = save_file(file, PROCESSED_FOLDER, 'original')

     # Load image and model
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Error reading image"}), 400
    model = load_model()

    # Process image for mask detection
    img = process_image(img, model)

    # Save processed image
    processed_filepath = save_file(file, PROCESSED_FOLDER, 'processed')
    cv2.imwrite(processed_filepath, img)

    return jsonify({"imageUrl": f'/{processed_filepath}'})

    
@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No selected video"}), 400

    filepath = save_file(video, PROCESSED_FOLDER, 'original')
    
    # Load video and model
    cap = cv2.VideoCapture(filepath)
    model = load_model()
    
    # Prepare output video writer
    processed_video_path = save_file(video, PROCESSED_FOLDER, 'processed')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Process frames and save
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_image(frame, model)
        out.write(frame)
    
    cap.release()
    out.release()
    
    return jsonify({"message": "Video processed and saved successfully", "output_file": f'/{processed_video_path}'})


if __name__ == '__main__':
    # with app.app_context():
    #     db.create_all()

    app.run(debug=True)