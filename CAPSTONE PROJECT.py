import numpy as np
import cv2

background = None
accumulated_weight = 0.5
roi_top = 30
roi_bottom = 300
roi_right = 300
roi_left = 600


def calc_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment(frame, threshold_min=25):
    diff = cv2.absdiff(background.astype("uint8"), frame)

    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)


def count_fingers(thresholded, hand_segment):
    # Ensure the contour is valid for convexity defects
    if len(hand_segment) < 5:
        return 0  # Not enough points to compute convexity defects

    conv_hull = cv2.convexHull(hand_segment)

    # Check for valid convex hull to prevent errors in defects calculation
    if len(conv_hull) < 3:
        return 0

    defects = cv2.convexityDefects(
        hand_segment, cv2.convexHull(hand_segment, returnPoints=False)
    )

    if defects is None:
        return 0

    fingers_count = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(hand_segment[s][0])
        end = tuple(hand_segment[e][0])
        far = tuple(hand_segment[f][0])

        # Calculate the distances between points
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        # Apply the cosine rule to find the angle
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

        # Check if the angle is less than 90 degrees
        if angle <= np.pi / 2:
            fingers_count += 1

    return fingers_count


cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    ret, frame = cam.read()

    frame_copy = frame.copy()

    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)

        if num_frames <= 59:
            cv2.putText(
                frame_copy,
                "WAIT, GETTING BACKGROUND",
                (200, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    else:
        hand = segment(gray)

        if hand is not None:
            thresholded, hand_segment = hand

            cv2.drawContours(
                frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 2
            )

            fingers = count_fingers(thresholded, hand_segment)

            cv2.putText(
                frame_copy,
                f"Fingers: {fingers}",
                (70, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            cv2.imshow("thresholded", thresholded)

    cv2.rectangle(
        frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 5
    )

    num_frames += 1

    cv2.imshow("finger count", frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
