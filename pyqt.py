import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp


import cv2
import pygame
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import google.generativeai as genai     # google ai
import numpy as np  
import math
from io import BytesIO
import time
from gtts import gTTS   





class HandGestureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0

        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.delay = 10
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Convert the frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands
            results = self.hands.process(frame_rgb)

            # Draw hand landmarks if hands are detected
            # if results.multi_hand_landmarks:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         # Draw hand landmarks on the frame
            self.draw_hand_landmarks(frame)

            # Convert the frame back to BGR for displaying in Tkinter
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def draw_hand_landmarks(self, frame):
        # h, w, _ = frame.shape
        # for landmark in landmarks.landmark:
        #     x, y = int(landmark.x * w), int(landmark.y * h)
        #     cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        GOOGLE_API_KEY = 'AIzaSyAY-gFnDMc1lsb0YKNAORmUsGejZ13gPsU'
        genai.configure(api_key=GOOGLE_API_KEY)

        model = genai.GenerativeModel('gemini-pro')

        # cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        # loaded_model = tf.keras.models.load_model('model.h5')
        classifier = Classifier("Model/Model_3/keras_model.h5", "Model/Model_3/labels.txt")
        offset = 20
        imgSize = 256
        counter = 0

        # labels = ["Hello", "I am fine", "No", "How", "Please", "Thank you", "Yes"]
        labels = ["Hello","How", "Iamfine","You","","Hold"]
        # Stores the update gestures
        Input = []
        s = ""
        flag = True
        prev = ""
        
        success, img = self.vid.read()
        # img = cv2.flip(img,1)
        # img = cv2.GaussianBlur(img, (15, 15), 0)
        imgOutput = img.copy()
        hands, img = detector.findHands(img,False)
        if hands:
            flag = True

            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            prediction, index = classifier.getPrediction(imgCrop, draw=False)
            print(prediction, index)

            # Inside the loop where you resize the cropped image
            # if aspectRatio > 1:
            #     k = imgSize / h
            #     wCal = math.ceil(k * w)
            #     # print("wCal:", wCal)
            #     imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            #     imgResizeShape = imgResize.shape
            #     # print("imgResizeShape:", imgResizeShape)
            #     wGap = math.ceil((imgSize - wCal) / 2)
            #     imgWhite[:, wGap: wCal + wGap] = imgResize
            #     imgWhite = cv2.resize(imgWhite, (256, 256))  # Resize to match model input shape
            #     # print("imgWhiteShape:", imgWhite.shape)
            #     prediction, index = classifier.getPrediction(imgWhite, draw=False)
            #     print(prediction, index)
            # else:
            #     k = imgSize / w
            #     hCal = math.ceil(k * h)
            #     # print("hCal:", hCal)
            #     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            #     imgResizeShape = imgResize.shape
            #     # print("imgResizeShape:", imgResizeShape)
            #     hGap = math.ceil((imgSize - hCal) / 2)
            #     imgWhite[hGap: hCal + hGap, :] = imgResize
            #     imgWhite = cv2.resize(imgWhite, (256, 256))  # Resize to match model input shape
            #     # print("imgWhiteShape:", imgWhite.shape)
            #     prediction, index = classifier.getPrediction(imgWhite, draw=False)
            #     print(prediction, index)

            if labels[-1] != labels[index]:
                Input.append(labels[index])

            cv2.rectangle(frame, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                        cv2.FILLED)

            cv2.putText(frame, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(frame, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # cv2.imshow('ImageCrop', imgCrop)
            # cv2.imshow('ImageWhite', imgWhite)
        else:
            if Input!=[]:

                if flag:
                    s = ""

                    for i in set(Input):
                        if i!="Hold":
                            s += i+" "
                    print(s)
                    # print(type(s)))
                    response = model.generate_content("correct the sentence without explanation : " + s)
                    print(response.text)
                    s = response.text
                    flag = False
                    # print(type(response))

                    Input = []
                    prev = s

                    tts = gTTS(text=s, lang='en')


                    audio_buffer = BytesIO()

                    tts.write_to_fp(audio_buffer)

                    audio_buffer.seek(0)

                    pygame.mixer.init()

                    pygame.mixer.music.load(audio_buffer)

                    pygame.mixer.music.play()

                    while pygame.mixer.music.get_busy():
                        time.sleep(1)

            cv2.putText(frame,prev,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        # cv2.imshow('Image', imgOutput)
        # cv2.waitKey(1)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the HandGestureApp class
HandGestureApp(tk.Tk(), "Hand Gesture Recognition with MediaPipe")
