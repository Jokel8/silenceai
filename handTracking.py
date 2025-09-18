from cvzone.HandTrackingModule import HandDetector
import cv2

capture = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()