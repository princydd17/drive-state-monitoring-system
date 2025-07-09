import speech_recognition as sr
import pandas as pd
import pygame
import concurrent.futures
from imutils import face_utils
import numpy as np
from gtts.tts import gTTS
import dlib
import cv2
import os
from threading import Thread
import time
from datetime import datetime, date
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

today = date.today()

# Project related imports for image processing

# Importing required files
import distraction_detection
import driver_alert

# Initialize pygame mixer for audio
pygame.mixer.init()

# Initializing the speech recognition
r = sr.Recognizer()

# A common speak function using pygame instead of playsound
def speak(filename, msg):
    try:
        tts = gTTS("{}".format("{}".format(msg)))
        tts.save('{}.mp3'.format(filename))
        pygame.mixer.music.load('{}.mp3'.format(filename))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        os.remove('{}.mp3'.format(filename))
    except Exception as e:
        print(f"Audio error: {e}")

# Function to play beep sound
def play_beep():
    try:
        pygame.mixer.music.load('beep.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"Beep error: {e}")

# The global name variable
NAME = ""

# Simplified name input to avoid microphone issues
NAME = "Driver"  # Default name
print("Using default name: Driver")
print("Started..")

# Declaring constants

# Drowsy Distance between eye lids threshold
EYE_OPEN_THERSHOLD = 7

# Drowsy Time Threshold
EYE_THRESH = 1

# Eye closed counter
EYE_CONSEC = 0

# Eye consec counter function flag
EYE_FLAG = False

# Frame capture delay
FPS = 1

# Loading the detector
detector = dlib.get_frontal_face_detector()  # type: ignore

# Loading the predictor
predictor = dlib.shape_predictor("eye_predictor.dat")  # type: ignore

# Distraction Counter
DISC_COUNTER = 0

# Distraction Counter Threshold
DISC_COUNT_THRES = 7

# Distraction Flag
DISC_FLAG = False
DISC_FLAG1 = False

# Taking video input form specified camera
cap = cv2.VideoCapture(0)

# Mouth Aspect Ratio Constant
MAR = 0
MAR_COUNTER = 0

# MAR Threshold
MAR_THRES = 29

# Alarm when driver is drowsy for a threshold value of frames
def distractionAlert(num, msg):
    try:
        tts = gTTS("{}".format(msg))
        tts.save('distractAlert.mp3')
        pygame.mixer.music.load('distractAlert.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        os.remove('distractAlert.mp3')
    except Exception as e:
        print(f"Distraction alert error: {e}")

# Multiple places flag
MULTIPLE_FACES_FLAG = False

# Alarm when multiple faces are detected
def multipleFacesAlarm(msg):
    global MULTIPLE_FACES_FLAG
    if MULTIPLE_FACES_FLAG:
        try:
            tts = gTTS("{}".format(msg))
            tts.save('multipleFaces.mp3')
            pygame.mixer.music.load('multipleFaces.mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            os.remove('multipleFaces.mp3')
            time.sleep(2)
        except Exception as e:
            print(f"Multiple faces alarm error: {e}")
        MULTIPLE_FACES_FLAG = False

# No Face detected flag
NO_FACE_FLAG = False
NO_FACE_FLAG1 = False
NO_FACE_THRES = 10

# Vars for df rows
Dfmsg = ""

NO_FACE_COUNT = 0

# Function for alarm when no face is detected
def NoFaceFunc():
    global NO_FACE_FLAG, NO_FACE_FLAG1, Dfmsg, NO_FACE_THRES, NO_FACE_COUNT
    time.sleep(NO_FACE_THRES)
    if not NO_FACE_FLAG and not NO_FACE_FLAG1:
        mst = []
        mst = Dfmsg.split(" | ")
        if mst[len(mst) - 1] != "No face detected" or NO_FACE_COUNT == 60:
            Dfmsg += "No face detected | "
            NO_FACE_COUNT = 0
        NO_FACE_FLAG = True
        try:
            tts = gTTS(
                "No face is detected please make sure you are properly visible in the camera")
            tts.save('noFace.mp3')
            pygame.mixer.music.load('noFace.mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            os.remove('noFace.mp3')
            play_beep()
        except Exception as e:
            print(f"No face alert error: {e}")

        # Beta mode
        # print(driver_alert.AdvanceDetection(cap))
        NO_FACE_FLAG = False

def increasingWithTime():
    global DISC_COUNTER, NO_FACE_COUNT
    time.sleep(1)
    DISC_COUNTER += 1
    NO_FACE_COUNT += 1
    if DISC_COUNTER > DISC_COUNT_THRES + 5:
        DISC_COUNTER = 0
    increasingWithTime()

incTimeThread = Thread(target=increasingWithTime)
incTimeThread.daemon = True
incTimeThread.start()

def eyeThresholdCount():
    time.sleep(1)
    global EYE_CONSEC, EYE_FLAG
    EYE_CONSEC += 1
    EYE_FLAG = False

data = {
    'Name': 'NEW SESSION',
    'Time': today.strftime("%d/%m/%Y") + " | " + datetime.now().strftime("%H:%M:%S"),
    'Message': '-----',
}

df = pd.DataFrame([data])

# To close flag
TO_END = False

while True:
    TO_END = True

    # Capturing frames
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Converting the camera input to gray scale image for detection
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Getting all detected faces rectangle
    rects = detector(gray, 1)
    if len(rects) == 0:
        NO_FACE_FLAG1 = False
        if not NO_FACE_FLAG:
            Thread(target=NoFaceFunc).start()
    else:
        NO_FACE_FLAG1 = True

    # Running required processes on all detected faces
    for (i, rect) in enumerate(rects):

        # Getting the face details from the predictor
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Distraction Check
        if DISC_COUNTER == DISC_COUNT_THRES and not DISC_FLAG1:
            DISC_FLAG1 = True
            print("Checking for distractions..")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(distraction_detection.Distraction, cap)
                DISC_FLAG, message, MAR = future.result()
            print(message)
            if MAR > MAR_THRES:
                MAR_COUNTER += 1
            if MAR_COUNTER >= 3:
                mm = "User is Speaking or talking on phone please focus on road"
                Thread(target=distractionAlert, args=(
                    10, "{}".format(mm))).start()
                Dfmsg += "{}".format(mm) + " | "
                MAR_COUNTER = 0
            if "Left" in message or "Right" in message:
                Thread(target=distractionAlert, args=(10, message)).start()
                Dfmsg += "{}".format(message) + " | "
        if DISC_FLAG:
            DISC_FLAG1 = False
            DISC_COUNTER = 0
            DISC_FLAG = False

        # Getting the coordinates of eyelids in required variables
        test = shape
        _, leyetop = test[1]
        _, leyebot = test[5]
        _, reyetop = test[7]
        _, reyebot = test[11]

        # Checking if the distance between eye lids is less than the
        # Given threshold
        if abs(leyetop - leyebot) < EYE_OPEN_THERSHOLD and abs(reyetop - reyebot) < EYE_OPEN_THERSHOLD:
            if not EYE_FLAG:
                eyeThresholdThread = Thread(target=eyeThresholdCount).start()
        else:
            EYE_CONSEC = 0

        # If eyes are closed for the entire threshold, performing specified event
        if(EYE_CONSEC >= EYE_THRESH):
            print("Drowsy")
            print(driver_alert.AdvanceDetection(cap))
            Dfmsg += "User is drowsy | "
            EYE_CONSEC = 0

        # Plotting the eye points on the screen
        for (x, y) in shape:
            cv2.circle(frame, (x, y), radius=2,
                       color=(0, 0, 255), thickness=-1)

    # Opening camera feed in a new window
    cv2.imshow(winname="Face", mat=frame)

    # Assigning ESC as closing key
    try:
        if cv2.waitKey(delay=1) == 27:
            break
    except Exception as e:
        print("Error while closing with ESC :" + str(e))
    if len(Dfmsg) != 0:
        Dfmsg = Dfmsg[:-2]
        new_row = pd.DataFrame([{
            'Name': NAME, 
            'Time': today.strftime("%d/%m/%Y") + " | " + datetime.now().strftime("%H:%M:%S"), 
            'Message': Dfmsg
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    Dfmsg = ""

# Closing the camera input and closing the windows
print(df)
cap.release()
cv2.destroyAllWindows()

df.to_csv('{}.csv'.format(NAME), mode='a', header=False)
# You're still reading? This is to take a look at what you want to double check your work. index_col = 0 will prevent a "Unnamed:0" column for appearing.
df_result = pd.read_csv('{}.csv'.format(
    NAME), index_col=0).reset_index(drop=True, inplace=True)
os.startfile('{}.csv'.format(NAME))
