from flask import Flask, render_template, jsonify, redirect
from flask_cors import CORS
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import time
import firebase_admin
from firebase_admin import credentials, storage
import threading

app = Flask(__name__)
CORS(app)
insightface = FaceAnalysis()
insightface.prepare(ctx_id=0, det_size=(640, 480))

# Load Firebase credentials and configure the app
cred = credentials.Certificate("crime.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'criminal-987d0.appspot.com'})
bucket = storage.bucket()

blobs = bucket.list_blobs(prefix='images/')
images = []
image_identifiers = []  # Initialize the list for image identifiers

for blob in blobs:
    if blob.name.endswith('.jpg') or blob.name.endswith('.png'):
        blob_data = blob.download_as_bytes()
        image = cv2.imdecode(np.frombuffer(blob_data, np.uint8), -1)
        if image is not None:
            if image.shape[2] == 4:  # Check if it has 4 channels (RGBA)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            images.append(image)
            image_identifiers.append(blob.name)  # Add the identifier/filename
        else:
            print(f"Failed to load image: {blob.name}")

reference_embeddings = []

for image in images:
    result = insightface.get(image)
    if result:
        reference_embedding = result[0]['embedding']
        reference_embeddings.append(reference_embedding)
    else:
        print('No face detected in an image')

def scan_faces(reference_embeddings, image_identifiers):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while time.time() - start_time < 20:  # Scanning for 10 seconds
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = insightface.get(rgb_frame)
            if result:
                frame_embedding = result[0]['embedding']
                match_found = False
                for i, reference_embedding in enumerate(reference_embeddings):
                    distance = cosine(reference_embedding, frame_embedding)
                    if distance < 0.6:
                        match_found = True
                        print(f"Matches to: {image_identifiers[i]}")  # Print the identifier
                        break  # Found a match, no need to continue checking
                if not match_found:
                    print('Not Matches')
            else:
                print('No face detected')
            cv2.imshow('Webcam', frame)
            cv2.waitKey(1)

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match_faces', methods=['POST'])
def start_face_matching():
    threading.Thread(target=scan_faces, args=(reference_embeddings, image_identifiers)).start()
    return jsonify({"message": "Face matching process started"})

@app.route('/hello')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
