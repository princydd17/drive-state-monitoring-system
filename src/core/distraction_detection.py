import cv2
import dlib
import numpy as np
from imutils import face_utils
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Loading the detector
detector = dlib.get_frontal_face_detector()  # type: ignore

# Loading the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # type: ignore

# Getting the points from the Facial landmarks
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]  # Points for jaw
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]  # Points for nose
# Points for left eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# Points for right eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

def Distraction(cap):
    # Capturing frames
    ret, frame = cap.read()
    if not ret:
        return False, "Failed to grab frame", 0

    # Converting the camera input to gray scale image for detection
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Getting all detected faces rectangle
    rects = detector(gray, 1)

    # Running required processes on all detected faces
    for (i, rect) in enumerate(rects):

        # Getting the face details from the predictor
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

        # Plotting the ellipses for both the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Getting the coordinates of eyelids in required variables
        test = shape
        _, leyetop = test[1]
        _, leyebot = test[5]
        _, reyetop = test[7]
        _, reyebot = test[11]

        # Checking if the distance between eye lids is less than the
        # Given threshold
        if abs(leyetop - leyebot) < 7 and abs(reyetop - reyebot) < 7:
            return True, "User is looking down", distance

        # Getting the coordinates of nose in required variables
        _, nosex = test[30]
        _, nosey = test[30]

        # Getting the coordinates of left eye in required variables
        _, lefteyex = test[36]
        _, lefteyey = test[36]

        # Getting the coordinates of right eye in required variables
        _, righteyex = test[45]
        _, righteyey = test[45]

        # Checking if the user is looking left
        if lefteyex < nosex - 20 and righteyex < nosex - 20:
            return True, "User is looking Left", distance

        # Checking if the user is looking right
        if lefteyex > nosex + 20 and righteyex > nosex + 20:
            return True, "User is looking Right", distance

        # Checking if the user is looking up
        if lefteyey < nosey - 20 and righteyey < nosey - 20:
            return True, "User is looking Up", distance

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box

        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), radius=2,
                       color=(0, 0, 255), thickness=-1)

    return False, "User is focused", 0
