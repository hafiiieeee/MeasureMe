import cv2
import mediapipe as mp
import time
import math
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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
KNOWN_WIDTH = 35  # Approximate shoulder width of an adult in cm
FOCAL_LENGTH = calculate_focal_length_in_pixels(RW, FOV)  # Focal length in pixels, calculated dynamically

# Set up the main tkinter window
root = tk.Tk()
root.title("Pose Estimation")

# Create a canvas to display the video feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Create labels for displaying measurements
distance_label = ttk.Label(root, text="Distance: ")
distance_label.pack()

length_label = ttk.Label(root, text="Length: ")
length_label.pack()

chest_label = ttk.Label(root, text="Chest: ")
chest_label.pack()

sleeve_label = ttk.Label(root, text="Half Sleeve: ")
sleeve_label.pack()

# Timing variables
time_interval = 8  # Time interval in seconds
start_time = time.time()  # Initial start time

# Start video capture
cap = cv2.VideoCapture(0)  # Use the default camera

def update_frame():
    global start_time  # Make sure to reference the global start_time variable

    success, image = cap.read()
    if not success:
        print("Failed to capture frame.")
        return

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

        # Update measurements every time the interval is passed
        current_time = time.time()
        if current_time - start_time >= time_interval:
            start_time = current_time  # Update the start time to current time

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

            # Update the labels with the new measurements
            distance_label.config(text=f"Distance: {distance:.2f} cm")
            length_label.config(text=f"Length: {Length:.2f} cm")
            chest_label.config(text=f"Chest: {Chest:.2f} cm")
            sleeve_label.config(text=f"Half Sleeve: {Sleeve:.2f} cm")

        # Draw landmarks, connections, and the center point
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.circle(image, center_point, 5, (255, 0, 0), -1)

    # Convert the image to PIL format for display in Tkinter
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Update the canvas with the new image
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    canvas.image = image_tk  # Keep a reference to avoid garbage collection

    # Continue updating the frame every 30ms
    root.after(30, update_frame)

# Start the video feed update loop
update_frame()

# Run the tkinter main loop
root.mainloop()

# Release the camera when the application closes
cap.release()
cv2.destroyAllWindows()
