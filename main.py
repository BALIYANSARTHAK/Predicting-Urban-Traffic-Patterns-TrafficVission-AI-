import tkinter as tk
from tkinter import filedialog
import cv2
import customtkinter as ctk
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


# Load model
model = YOLO("yolov8n.pt")   # Replace with trained model if you have

# Root window
window = ctk.CTk()
window.geometry("1200x700")
window.title("TrafficVision AI")

# Global vehicle count
vehicle_count = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}

# ----------------- FUNCTIONS -----------------

def upload_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video/Images", "*.mp4;*.avi;*.mov;*.jpg;*.png")]
    )
    if file_path:
        process_file(file_path)

def process_file(file_path):
    global vehicle_count
    vehicle_count = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}

    if file_path.endswith((".jpg", ".png")):  # Image
        img = cv2.imread(file_path)
        results = model(img)
        for r in results[0].boxes.data:
            cls_id = int(r[5])
            cls_name = model.names[cls_id]
            if cls_name in vehicle_count:
                vehicle_count[cls_name] += 1
        cv2.imshow("Result", results[0].plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:  # Video
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            for r in results[0].boxes.data:
                cls_id = int(r[5])
                cls_name = model.names[cls_id]
                if cls_name in vehicle_count:
                    vehicle_count[cls_name] += 1
            cv2.imshow("Result", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    print("Vehicle count:", vehicle_count)

def graphframe():
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(vehicle_count.keys(), vehicle_count.values(), color="skyblue")
    ax.set_title("Vehicle Count")
    ax.set_ylabel("Count")

    # Embed graph in Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

# ----------------- UI -----------------
ctk.CTkButton(window, text="Upload File", command=upload_file).pack(pady=20)
ctk.CTkButton(window, text="Show Graph Data", command=graphframe).pack(pady=20)

# Run app
window.mainloop()
