import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0) # Use "0" for webcam or replace with video file path
# cap.set(3, 640) # Set width
# cap.set(4, 480) # Set height
segmentor = SelfiSegmentation()

while (True):
    success, img = cap.read()
    if not success:
        print("No more stream :(")
        break
    imgOut = segmentor.removeBG(img, (255, 255, 255), cutThreshold=0.1) # Play with Threshold if Hands are cuted out
    cv2.imshow("Image", img) # Raw image for comparison
    cv2.imshow("Image OUT", imgOut) # No background
    if cv2.waitKey(1) == ord('q'): # Press "q" to quit
        break
    
cap.release()
cv2.destroyAllWindows()