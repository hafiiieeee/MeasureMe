import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to calculate focal length in pixels from the field of view and resolution width
def calculate_focal_length_in_pixels(resolution_width, FOV_degrees):
    FOV_radians = FOV_degrees * (math.pi / 180)
    focal_length_pixels = resolution_width / (2 * math.tan(FOV_radians / 2))
    return focal_length_pixels

# Function to estimate horizontal offset and distance from the camera using shoulder landmarks
def estimate_position_and_distance(image_width, image_height, landmark1, landmark2, focal_length, known_width):
    # Calculate the pixel distance between landmarks
    pixel_distance = math.hypot((landmark1.x - landmark2.x) * image_width, (landmark1.y - landmark2.y) * image_height)

    # Avoid division by zero (if pixel distance is too small)
    if pixel_distance == 0:
        return 0, 0, (0, 0)

    # Calculate distance from the camera
    distance = (known_width * focal_length) / pixel_distance

    # Calculate average position (central point between the shoulders in image coordinates)
    avg_x = (landmark1.x + landmark2.x) / 2 * image_width
    avg_y = (landmark1.y + landmark2.y) / 2 * image_height

    # Calculate horizontal offset from the center of the image in pixels
    center_x = image_width / 2
    pixel_offset = avg_x - center_x

    # Assuming a constant real-world field of view, calculate offset in cm (approximation)
    distance_per_pixel = math.tan(math.radians(FOV / 2)) * 2 * distance / image_width
    cm_offset = pixel_offset * distance_per_pixel

    return cm_offset, distance, (int(avg_x), int(avg_y))

# Constants provided by the user
FOV = 60  # Field of View in degrees
RW = 1280  # Resolution Width in pixels

# Constants
KNOWN_WIDTH = 35  # Approximate shoulder width of an adult in cm
FOCAL_LENGTH = calculate_focal_length_in_pixels(RW, FOV)  # Focal length in pixels, calculated dynamically

# List to store X and Y coordinates
xy_coordinates = []

# Counter for instances recorded
instances_recorded = 0
max_instances = 8  # Maximum number of instances to record

# Timing variables
start_time = time.time()
time_interval = 8  # Time interval in seconds

# Capture video
cap = cv2.VideoCapture(0)  # Use the default camera

while cap.isOpened() and instances_recorded < max_instances:
    success, image = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    # Convert to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Extracting relevant landmarks (left and right shoulders, right hip, right elbow)
        landmark1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        landmark2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        landmark3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        landmark4 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        ih, iw, ic = image.shape
        cm_offset, distance, center_point = estimate_position_and_distance(iw, ih, landmark1, landmark2, FOCAL_LENGTH, KNOWN_WIDTH)

        if time.time() - start_time >= time_interval:
            x_coord = cm_offset
            y_coord = distance
            xy_coordinates.append((x_coord, y_coord))
            instances_recorded += 1
            start_time = time.time()

        # Calculate sleeve, length, and chest distances
        sleeve = math.sqrt(((landmark2.x - landmark4.x) ** 2) + ((landmark2.y - landmark4.y) ** 2))
        length = math.sqrt(((landmark2.x - landmark3.x) ** 2) + ((landmark2.y - landmark3.y) ** 2))
        chest = math.sqrt(((landmark2.x - landmark1.x) ** 2) + ((landmark2.y - landmark1.y) ** 2))

        # Distance-based scaling factor for measurements
        if distance <= 100:
            d = 10
        elif distance <= 200:
            d = 20
        elif distance <= 300:
            d = 30
        elif distance <= 400:
            d = 40
        elif distance <= 500:
            d = 50
        else:
            print("COME CLOSE")

        Sleeve = ((sleeve * d) * (distance / d)) / 10
        Length = ((length * d) * (distance / d)) / 5
        Chest = ((chest * d) * (distance / d)) / 5

        # Data to be displayed in the table
        data = [
            ["Measurement", "Measure(in cm)"],
            ["Distance", f"{distance:.2f} cm"],
            ["Length", f"{Length:.2f} cm"],
            ["Chest", f"{Chest:.2f} cm"],
            ["Half sleeve", f"{Sleeve:.2f} cm"]
        ]

        # Draw landmarks, connections, and the center point
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.circle(image, center_point, 5, (255, 0, 0), -1)
        cv2.putText(image, "STAND PROPERLY INFRONT OF CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the table on the video frame
        table_top_left = (50, 50)
        table_top_right = (350, 50)
        table_bottom_left = (50, 300)
        table_bottom_right = (350, 300)
        cv2.rectangle(image, table_top_left, table_bottom_right, (255, 255, 255), 2)

        for i in range(len(data) + 1):
            cv2.line(image, (table_top_left[0], table_top_left[1] + i * 50), (table_top_right[0], table_top_right[1] + i * 50), (255, 255, 255), 2)

        cv2.line(image, (table_top_left[0] + 150, table_top_left[1]), (table_bottom_left[0] + 150, table_bottom_left[1]), (255, 255, 255), 2)

        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                cv2.putText(image, cell, (table_top_left[0] + j * 150 + 10, table_top_left[1] + i * 50 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('MediaPipe Pose', image)  # Directly show the updated image with the table

    # Exit the loop when pressing the "Esc" key
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
