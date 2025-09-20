# configurable camera index
CAMERA_INDEX = 0

from cvzone.HandTrackingModule import HandDetector
import cv2

# Initialize the webcam to capture video
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Kein Stream: Kamera konnte nicht geöffnet werden.")
    exit()

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

try:
    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        success, img = cap.read()

        # If frame capture failed, break
        if not success or img is None:
            print("Kein Kamerabild empfangen.")
            break

        # Find hands in the current frame
        hands, img = detector.findHands(img, draw=True, flipType=True)

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 = hands[0]
            lmList1 = hand1.get("lmList", [])
            bbox1 = hand1.get("bbox", None)
            center1 = hand1.get('center', None)
            handType1 = hand1.get("type", None)

            # Count the number of fingers up for the first hand
            fingers1 = detector.fingersUp(hand1)
            print(f'H1 = {fingers1.count(1)}', end=" ")

            # Calculate distance between specific landmarks on the first hand (index and middle finger) if available
            if len(lmList1) > 12:
                p_idx = tuple(lmList1[8][0:2])
                p_mid = tuple(lmList1[12][0:2])
                length, info, img = detector.findDistance(p_idx, p_mid, img, color=(255, 0, 255), scale=10)

            # Check if a second hand is detected
            if len(hands) >= 2:
                hand2 = hands[1]
                lmList2 = hand2.get("lmList", [])
                bbox2 = hand2.get("bbox", None)
                center2 = hand2.get('center', None)
                handType2 = hand2.get("type", None)

                # Count the number of fingers up for the second hand
                fingers2 = detector.fingersUp(hand2)
                print(f'H2 = {fingers2.count(1)}', end=" ")

                # Calculate distance between the index fingers of both hands if available
                if len(lmList1) > 8 and len(lmList2) > 8:
                    p1 = tuple(lmList1[8][0:2])
                    p2 = tuple(lmList2[8][0:2])
                    length, info, img = detector.findDistance(p1, p2, img, color=(255, 0, 0), scale=10)

            print(" ")  # New line for better readability of the printed output

        # Display the image in a window
        cv2.imshow("Image", img)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q to exit
            break
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()