from keras.models import load_model
import cv2
import numpy as np
import time
import pyttsx3
import os

model = load_model("MyDataset/keras_model.h5", compile=False)
class_names = open("MyDataset/labels.txt", "r").readlines()
camera = cv2.VideoCapture(1)

speak = pyttsx3.init()

sentence = [""]
os.system('cls' if os.name == 'nt' else 'clear') 

while True:
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1


    prediction = model.predict(image, verbose=0) # remove unncessary prints
    index = np.argmax(prediction)
    class_name = str((class_names[index])[2:]).replace("\n", "") # get similarity as a string
    confidence_score = prediction[0][index] # % similarity


    if(class_name != "background"): # update the print on the console for the sentence and read it out
        if(sentence[-1] != class_name):
            sentence.append(class_name)
        os.system('cls' if os.name == 'nt' else 'clear') 
        print((' '.join(sentence))[1:])
        if(sentence[-1] != "."):
            speak.say(class_name)
            speak.runAndWait()


    time.sleep(0.5) # fps captured

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()