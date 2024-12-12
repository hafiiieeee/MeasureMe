import cv2
import mediapipe as mp
import time
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# MediaPipe Pose Initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to calculate focal length in pixels
def calculate_focal_length(resolution_width, FOV_degrees):
    FOV_radians = FOV_degrees * (math.pi / 180)
    return resolution_width / (2 * math.tan(FOV_radians / 2))


# Function to estimate position and distance
def estimate_position_and_distance(image_width, image_height, landmark1, landmark2, focal_length, known_width):
    pixel_distance = math.hypot((landmark1.x - landmark2.x) * image_width, (landmark1.y - landmark2.y) * image_height)

    if pixel_distance == 0:
        return 0, 0, (0, 0)

    distance = (known_width * focal_length) / pixel_distance
    avg_x = (landmark1.x + landmark2.x) / 2 * image_width
    avg_y = (landmark1.y + landmark2.y) / 2 * image_height

    center_x = image_width / 2
    pixel_offset = avg_x - center_x
    distance_per_pixel = math.tan(math.radians(FOV / 2)) * 2 * distance / image_width
    cm_offset = pixel_offset * distance_per_pixel

    return cm_offset, distance, (int(avg_x), int(avg_y))


# Constants
FOV = 60  # Field of View in degrees
RESOLUTION_WIDTH = 1280  # Camera resolution width
KNOWN_SHOULDER_WIDTH = 35  # Shoulder width in cm
FOCAL_LENGTH = calculate_focal_length(RESOLUTION_WIDTH, FOV)  # Focal length in pixels


# Home Page Class
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="black")

        # Title
        title_label = ttk.Label(self, text="FIND YOUR MEASUREMENT", font=("Arial", 24, "bold"), background="black", foreground="white")
        title_label.pack(pady=100)

        # Clothing Type Dropdown
        self.clothing_options = ["Shirt", "T-shirt", "Hoodie"]
        self.selected_clothing = tk.StringVar()
        self.selected_clothing.set(self.clothing_options[0])  # Default to Shirt
        clothing_dropdown = ttk.OptionMenu(self, self.selected_clothing, *self.clothing_options)
        clothing_dropdown.pack(pady=20)

        # Start Button
        start_button = ttk.Button(self, text="START", command=lambda: controller.show_frame(PoseEstimationPage))
        start_button.pack(pady=20)


# Pose Estimation Page Class
class PoseEstimationPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Video feed canvas
        self.canvas = tk.Canvas(self, width=640, height=480)
        self.canvas.pack()

        # Labels for measurements
        self.distance_label = ttk.Label(self, text="Distance: ")
        self.distance_label.pack()
        self.length_label = ttk.Label(self, text="Length: ")
        self.length_label.pack()
        self.chest_label = ttk.Label(self, text="Chest: ")
        self.chest_label.pack()
        self.sleeve_label = ttk.Label(self, text="Half Sleeve: ")
        self.sleeve_label.pack()

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.time_interval = 8  # Time interval in seconds
        self.update_frame()

    def update_frame(self):
        success, image = self.cap.read()
        if not success:
            print("Failed to capture frame.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            ih, iw, ic = image.shape
            landmarks = results.pose_landmarks.landmark

            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            cm_offset, distance, center_point = estimate_position_and_distance(
                iw, ih, l_shoulder, r_shoulder, FOCAL_LENGTH, KNOWN_SHOULDER_WIDTH
            )

            # Time interval to update measurements
            current_time = time.time()
            if current_time - self.start_time >= self.time_interval:
                self.start_time = current_time

                # Calculate sleeve, length, chest
                sleeve = math.sqrt(
                    ((r_shoulder.x - r_elbow.x) ** 2) + ((r_shoulder.y - r_elbow.y) ** 2)
                )
                length = math.sqrt(
                    ((r_shoulder.x - r_hip.x) ** 2) + ((r_shoulder.y - r_hip.y) ** 2)
                )
                chest = math.sqrt(
                    ((l_shoulder.x - r_shoulder.x) ** 2) + ((l_shoulder.y - r_shoulder.y) ** 2)
                )

                # Scaling factor for measurements
                scaling_factor = max(10, min(50, int(distance // 100) * 10))
                Sleeve = ((sleeve * scaling_factor) * (distance / scaling_factor)) / 10
                Length = ((length * scaling_factor) * (distance / scaling_factor)) / 5
                Chest = ((chest * scaling_factor) * (distance / scaling_factor)) / 5

                # Update labels
                self.distance_label.config(text=f"Distance: {distance:.2f} cm")
                self.length_label.config(text=f"Length: {Length:.2f} cm")
                self.chest_label.config(text=f"Chest: {Chest:.2f} cm")
                self.sleeve_label.config(text=f"Half Sleeve: {Sleeve:.2f} cm")

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.circle(image, center_point, 5, (255, 0, 0), -1)

        # Convert image for Tkinter
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tk = ImageTk.PhotoImage(image_pil)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.canvas.image = image_tk

        # Refresh frame every 30ms
        self.after(30, self.update_frame)

    def on_close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# Main Application Class
class MeasureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FIND YOUR MEASUREMENT")
        self.geometry("800x600")

        # Container to hold all pages
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Dictionary to hold pages
        self.frames = {}

        for Page in (HomePage, PoseEstimationPage):
            frame = Page(container, self)
            self.frames[Page] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(HomePage)

    def show_frame(self, page_class):
        frame = self.frames[page_class]
        frame.tkraise()


# Run the App
if __name__ == "__main__":
    app = MeasureApp()
    app.mainloop()
