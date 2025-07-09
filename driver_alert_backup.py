# Basic python imports
import threading
import time
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Imports related to image processing and alarms
import cv2
import dlib
import numpy as np
from imutils import face_utils
import pygame
from scipy.spatial import distance as dist

# Initialize pygame mixer for audio
pygame.mixer.init()

# Declaring Required Constants And Thresholds

# EAR ratio for drowsy detection
EYE_AR_THRESH = 0.25

# Threshold for blink detection
EYE_BLINK_THRESH = 0.2

# Time threshold for drowsy detection
EYE_AR_CONSEC_FRAMES = 4

# Drowsy Detection seconds  counter ( eyes closed for sec )
COUNTER = 0

# Flag for alarm
ALARM_ON = False

# Flag for alarm off
ALARM_OFF = True

# Flag required for various alarms
alarm_status = False
alarm_status2 = False
saying = False

# Getting the points from the Facial landmarks
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]  # Points for jaw
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]  # Points for nose
# Points for left eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# Points for right eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Thread kill flag
THREAD_KILL_FLAG = False

# Increasing values flag
INC_FLAG = False

# Increasing values per second
def increasingWithTime():
    global COUNTER
    time.sleep(1)
    COUNTER += 1
    increasingWithTime()

# Function to play beep sound
def play_beep():
    try:
        pygame.mixer.music.load('beep.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"Beep error: {e}")

# Alarm when driver is drowsy for a threshold value of frames
def drowsyAlert(num):
    global THREAD_KILL_FLAG
    while not ALARM_OFF:
        play_beep()
        for _ in range(5):
            if ALARM_OFF:
                return "Exiting"
            time.sleep(1)
        if THREAD_KILL_FLAG:
            THREAD_KILL_FLAG = False
            break

# EAR formula function, to calculate the distance between eyelids
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    # Calculating the EAR from the standard formula
    EAR = (A + B) / (2.0 * C)
    return EAR

# MAR formula function, to calculate the distance between lips
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Load the detector
detector = dlib.get_frontal_face_detector()  # type: ignore

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # type: ignore

# Flag for Blink Check
blinkCheck = False

def AdvanceDetection(cap):
    # Drowsy alarm thread
    DROWSY_ALARM_THREAD = threading.Thread(target=drowsyAlert, args=(10,))

    global blinkCheck, predictor, detector, EYE_AR_THRESH
    global EYE_AR_CONSEC_FRAMES, EYE_BLINK_THRESH
    global COUNTER, ALARM_ON
    global alarm_status, alarm_status2, saying, ALARM_OFF
    global THREAD_KILL_FLAG, INC_FLAG
    COUNTER = 0
    while True:
        # taking frames from the video
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert image into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find rectangle box for all detected faces
        rects = detector(gray, 1)

        # For iterating through all the faces detected by detector
        for (i, rect) in enumerate(rects):

            # determine the facial landmarks for the face region
            shape = predictor(gray, rect)

            # the four corners of rectangle around face
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # Displaying the rectangle on the screen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            shape = face_utils.shape_to_np(shape)

            # Plotting eyes from the points taken from facial landmark
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            distance = lip_distance(shape)

            # Getting EAR for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # Plotting the ellipses for both the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Checking if the driver is drowsy from the given threshold value
            if ear < EYE_AR_THRESH:
                if INC_FLAG:
                    incTimeThread = threading.Thread(target=increasingWithTime)
                    incTimeThread.daemon = True
                    incTimeThread.start()
                    INC_FLAG = False
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_OFF = False
                        try:
                            DROWSY_ALARM_THREAD.start()
                        except Exception as e:
                            print("Error in start while starting thread: ", e)
                        ALARM_ON = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False
                ALARM_OFF = True
                INC_FLAG = True
                try:
                    pass
                except Exception as e:
                    print("Error in start while join : " + str(e))
                return "User is awake now"

            # Printing the current EAR of the driver on screen
            cv2.putText(
                frame,
                "EAR: {:.2f}".format(ear),
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

            # Printing the current MAR of the driver on screen
            cv2.putText(
                frame,
                "MAR: {:.2f}".format(distance),
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box

            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), radius=2,
                           color=(0, 0, 255), thickness=-1)

        # show the image
        cv2.imshow(winname="Face", mat=frame)

        # Exit when escape is pressed
        if cv2.waitKey(delay=1) == 27:
            break

    # When everything done, release the video capture and video write objects
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    AdvanceDetection(cap)
