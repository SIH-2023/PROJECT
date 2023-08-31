import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
ip_camera = "http://192.168.137.244:8080/video"
cap = cv2.VideoCapture(ip_camera)


LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_HAND = 13
RIGHT_HAND = 14


RUNNING_THRESHOLD = 30  # Adjust this value based on your needs
CRAWLING_THRESHOLD = 10  # Adjust this value based on your needs
STANDING_THRESHOLD = 20  # Adjust this value based on your needs
WALKING_THRESHOLD = 15  # Adjust this value based on your needs
JUMPING_THRESHOLD = 40  # Adjust this value based on your needs

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Get the y-coordinates of the ankles, hips, knees, and hands
            left_ankle_y = results.pose_landmarks.landmark[LEFT_ANKLE].y * image.shape[0]
            right_ankle_y = results.pose_landmarks.landmark[RIGHT_ANKLE].y * image.shape[0]
            left_knee_y = results.pose_landmarks.landmark[LEFT_KNEE].y * image.shape[0]
            right_knee_y = results.pose_landmarks.landmark[RIGHT_KNEE].y * image.shape[0]
            left_hand_y = results.pose_landmarks.landmark[LEFT_HAND].y * image.shape[0]
            right_hand_y = results.pose_landmarks.landmark[RIGHT_HAND].y * image.shape[0]

            # Check for running pose
            if left_ankle_y > RUNNING_THRESHOLD and right_ankle_y > RUNNING_THRESHOLD:
                cv2.putText(image, "Running", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check for crawling pose
            if left_knee_y > CRAWLING_THRESHOLD and right_knee_y > CRAWLING_THRESHOLD:
                cv2.putText(image, "Crawling", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check for standing pose
            if (
                abs(left_ankle_y - right_ankle_y) < STANDING_THRESHOLD
                and abs(left_knee_y - right_knee_y) < STANDING_THRESHOLD
            ):
                cv2.putText(image, "Standing", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check for walking pose
            if (
                abs(left_ankle_y - right_ankle_y) < WALKING_THRESHOLD
                and abs(left_hand_y - right_hand_y) < WALKING_THRESHOLD
            ):
                cv2.putText(image, "Walking", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Check for jumping pose
            if abs(left_hand_y - right_hand_y) > JUMPING_THRESHOLD:
                cv2.putText(image, "Jumping", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
