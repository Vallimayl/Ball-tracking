import cv2
import numpy as np

#importing kalman filter class
class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        #''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
    #returning the predicted x and y co-ordinates
#importing a class to detect the ball
class BallDetector:
    def __init__(self):
        # Create mask for blue color
        self.low_ball = np.array([11, 128, 90])
        self.high_ball = np.array([179, 255, 255])

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks with color ranges
        mask = cv2.inRange(hsv_img, self.low_ball, self.high_ball)

        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (0, 0, 0, 0)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            box = (x, y, x + w, y + h)
            break
#returning the contour's dimensions
        return box
#calling balldetector and kalman filter classes
bd=BallDetector()
kf=KalmanFilter()
cap=cv2.VideoCapture(r'C:\Users\spval\Downloads\Adv-IP.mp4')
#if video is available/can be read, enter into while loop
while True:
    #reading the cap and assigning the video to frame
    tf,frame=cap.read()
    #if true false(video is not running/ finished)  is false, come out of while loop
    if tf is False:
        break
    #getting the contour's dimension from frame/video
    dim=bd.detect(frame)
    x,y,x1,y1=dim
    #getting centres' x,y co-ordinates
    cx=int((x+x1)/2)
    cy=int((y+y1)/2)
    #predicting the next position by using the current position
    predicted=kf.predict(cx,cy)
    #drawing a circle based on current position of the ball
    cv2.circle(frame, (cx,cy),20,(0,0,255),-1)
    #drawing another circle based on the predicted position of the ball
    cv2.circle(frame,(predicted[0],predicted[1]),20,(255,0,0),-1)
    #showing the frame in while loop as we're using frame only inside the while loop
    #if we use frame outside while loop, there will be no frame, so it will show error as width and height of frame
    #should be greater than zero
    cv2.imshow('frame',frame)
    #waitKey 1 to display it as video
    #i got struck while using this waitkey function outside while loop
    cv2.waitKey(1)
    #always blue(predicted) ball will be ahead in trajectory when compared to red(current position) ball as the
    #former is the one which is predicted
    #after the video gets over, it will get break from while loop because of the if loop, so we can close the 
    #displayed frame window




