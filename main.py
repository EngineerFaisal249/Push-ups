import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import math

# Initialize push-up counter and direction
counter = 0
direction = 0

# Load video file
cap = cv2.VideoCapture(r'vid1.mp4')

# Initialize Pose Detector
pd = PoseDetector(trackCon=0.70, detectionCon=0.70)

def angles(lmlist, p1, p2, p3, p4, p5, p6, drawpoints):
    global counter
    global direction

    if len(lmlist) != 0:
        # Extract points directly
        x1, y1 = lmlist[p1][1], lmlist[p1][2]
        x2, y2 = lmlist[p2][1], lmlist[p2][2]
        x3, y3 = lmlist[p3][1], lmlist[p3][2]
        x4, y4 = lmlist[p4][1], lmlist[p4][2]
        x5, y5 = lmlist[p5][1], lmlist[p5][2]
        x6, y6 = lmlist[p6][1], lmlist[p6][2]

        if drawpoints:
            # Draw key points and lines
            for (x, y) in [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]:
                cv2.circle(img, (x, y), 10, (255, 0, 255), 5)
                cv2.circle(img, (x, y), 15, (0, 255, 0), 5)
            for (start, end) in [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)),
                                 ((x4, y4), (x5, y5)), ((x5, y5), (x6, y6)),
                                 ((x1, y1), (x4, y4))]:
                cv2.line(img, start, end, (0, 0, 255), 6)

        # Calculate angles for left and right hands
        lefthandangle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                     math.atan2(y1 - y2, x1 - x2))
        righthandangle = math.degrees(math.atan2(y6 - y5, x6 - x5) -
                                      math.atan2(y4 - y5, x4 - x5))

        # Normalize angles to a scale of 0-100
        leftHandAngle = int(np.interp(lefthandangle, [-30, 180], [100, 0]))
        rightHandAngle = int(np.interp(righthandangle, [34, 173], [100, 0]))

        left, right = leftHandAngle, rightHandAngle

        # Update counter and direction based on angles
        if left >= 70 and right >= 70:
            if direction == 0:
                counter += 0.5
                direction = 1
        if left <= 70 and right <= 70:
            if direction == 1:
                counter += 0.5
                direction = 0

        # Draw the counter
        cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
        cv2.putText(img, str(int(counter)), (20, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 7)

        # Draw progress bars for left and right arms
        leftval = np.interp(left, [0, 100], [400, 200])
        rightval = np.interp(right, [0, 100], [400, 200])

        cv2.putText(img, 'R', (24, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
        cv2.rectangle(img, (8, 200), (50, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (8, int(rightval)), (50, 400), (255, 0, 0), -1)

        cv2.putText(img, 'L', (962, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
        cv2.rectangle(img, (952, 200), (995, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (952, int(leftval)), (995, 400), (255, 0, 0), -1)

        # Red progress bar if angles exceed 70
        if left > 70:
            cv2.rectangle(img, (952, int(leftval)), (995, 400), (0, 0, 255), -1)

        if right > 70:
            cv2.rectangle(img, (8, int(rightval)), (50, 400), (0, 0, 255), -1)

# Main loop
while True:
    ret, img = cap.read()
    if not ret:
        cap = cv2.VideoCapture(r'vid1.mp4')
        continue

    img = cv2.resize(img, (1000, 500))
    cvzone.putTextRect(img, 'AI Push Up Counter', [345, 30], thickness=2, border=2, scale=2.5)

    # Detect pose and landmarks
    pd.findPose(img, draw=False)
    lmlist, bbox = pd.findPosition(img, draw=False, bboxWithHands=False)

    # Calculate angles and draw overlays
    angles(lmlist, 11, 13, 15, 12, 14, 16, drawpoints=True)

    # Display the frame
    cv2.imshow('Push-Up Counter', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
