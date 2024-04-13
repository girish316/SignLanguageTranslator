from flask import Flask, render_template, Response, jsonify
import cv2
from keras.models import load_model
import numpy as np
import time
import pyttsx3
import threading

model = load_model("MyDataset/keras_model.h5", compile=False)
class_names = open("MyDataset/labels.txt", "r").readlines()
confidence = ""
camera = cv2.VideoCapture()

sentence = [""]

app = Flask(__name__)
camera = cv2.VideoCapture(1)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processImage(frame)
                    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.3)

            
def processImage(frame):
    global confidence
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    class_name = str((class_names[index])[2:]).replace("\n", "")
    confidence_score = prediction[0][index]
    confidence = str(np.round(confidence_score * 100))[:-2]

    if class_name != "background":
        if sentence[-1] != class_name:
            sentence.append(class_name)
            thread = threading.Thread(target=say, args=(class_name,))
            thread.start()

def say(text):
    speak = pyttsx3.init()
    speak.say(text)
    speak.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def sentence_feed():
    data = (' '.join(sentence))[1:]
    return jsonify(data=data)

@app.route('/sign')
def sign_feed():
    data = ""
    if sentence[-1] != "" or (confidence == "100" and sentence[-1] == "."):   
        data = (confidence+"% similar to "+sentence[-1])
    return jsonify(data=data)

if __name__ == "__main__":
    app.run(debug=True)
