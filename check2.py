from scipy.spatial import distance as dist
import numpy as np
import imutils
import time
import dlib
import cv2

blinkThreshold = 0.15
video_in = "/media/sf_ShareFolder/speech1.mp4"
#video_in = "driving_car.avi"
video_out = "output.avi"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

camera = cv2.VideoCapture(video_in)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(video_out,fourcc, 30.0, (int(width),int(height)))

def eye_aspect_ratio(eye):
        print(eye)
        dist1 = dist.euclidean(eye[1], eye[5])
        dist2 = dist.euclidean(eye[2], eye[4])
        dist3 = dist.euclidean(eye[3], eye[0])

        EAR = (dist1 + dist2) / (2.0 * dist3)

        return EAR

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image

def getEyeShapes(landmarks):
    #right eye: 36~41
    #left eye: 42~47
    right = []
    left = []

    for id in range(42,48):
        left.append((landmarks.part(id).x, landmarks.part(id).y))

    for id in range(36,42):
        right.append((landmarks.part(id).x, landmarks.part(id).y))

    return np.array(left), np.array(right)

def drawEyeHull(leftEye, rightEye, img):
    left = cv2.convexHull(leftEye)
    right = cv2.convexHull(rightEye)
    cv2.drawContours(img, [left], -1, (0, 255, 0), 1)
    cv2.drawContours(img, [right], -1, (0, 255, 0), 1)

    return img

def bbox_2_img(img, bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    return img[y:y+h, x:x+w]


frameID = 0
bgDark_w = 0
blinkCount = 0
bgImage = np.zeros((height,width+bgDark_w,3), np.uint8)
bgImage[:,0:width+bgDark_w] = (0,0,0)      # (B, G, R)
print(bgImage.shape)
#bgImage[:,0.5*width:width] = (0,255,0)

while True:
    frameBG = bgImage.copy()
    (grabbed, frame) = camera.read()

    #frame = imutils.resize(frame, width=800)
    #frameBG = imutils.resize(frameBG, width=800)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)  #Detect faces use Dlib
    leftEyes = []
    rightEyes = []

    print("Face detected: ",len(rects))
    for faceid, rect in enumerate(rects):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #faceid, rect.left(), rect.top(), rect.right(), rect.bottom()))

        shape = predictor(gray, rect)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

        leftEye, rightEye = getEyeShapes(shape)
        leftEyes.append(leftEye)
        rightEyes.append(rightEye)
        #print("left: {}, right: {} ...".format(leftEye, rightEye))

    #print("LeftEyes:", leftEyes)
    if(len(leftEyes)>0 and len(rightEyes)>0):
        leftEAR = eye_aspect_ratio(leftEyes[0])
        rightEAR = eye_aspect_ratio(rightEyes[0])
        blinkEAR = (leftEAR + rightEAR) / 2

        blinkYN = "False"
        if(blinkEAR<=blinkThreshold):
            blinkCount += 1
            blinkYN = "True"

        line1 = "Left   Right   Blink"
        line2_a = str(round(leftEAR,2))
        line2_b = str(round(rightEAR,2))
        line2_c = blinkYN

        frameBG[0:height, bgDark_w:width+bgDark_w] = frame
        frameBG = putText(frameBG, line1, 20, 140, (0,255,0), 2, 1.2)
        frameBG = putText(frameBG, line2_a, 20, 200, (0,255,0), 2, 1.2)
        frameBG = putText(frameBG, line2_b, 150, 200, (0,255,0), 2, 1.2)
        frameBG = putText(frameBG, line2_c, 300, 200, (0,255,0), 2, 1.2)

        frame = drawEyeHull(leftEyes[0], rightEyes[0], frame)
        bbox_left = cv2.boundingRect(leftEyes[0])
        bbox_right = cv2.boundingRect(rightEyes[0])

        img_left = imutils.resize(bbox_2_img(frame, bbox_left), width = 150)
        img_right = imutils.resize(bbox_2_img(frame, bbox_right), width = 150)

        frameBG[50:50+img_left.shape[0], 40:40+img_left.shape[1]] = img_left
        frameBG[50:50+img_right.shape[0], 220:220+img_right.shape[1]] = img_right


    cv2.imshow("FRAME", imutils.resize(frameBG, width=850))
    cv2.imwrite("/media/sf_ShareFolder/snapshot.png", frameBG)
    frameID  += 1

    print("Frame: #", frameID )
    cv2.waitKey(1)
