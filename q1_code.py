import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- Import dependencies ---
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- Initialize YOLO model ---
try:
    model = YOLO("runs/detect/train/weights/best.pt")
except Exception:
    print("⚠️ best.pt not found — using untrained YOLO model.")
    model = YOLO()  # Initialize default YOLO model (no weights)

# License plate damage classes
classes = ['broken', 'non broken']

# Function to perform image inference
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return None

    results = model.predict(source=image_path, verbose=False)
    output = results[0]

    for box, conf, cls in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        if class_id < len(classes):
            class_name = classes[class_id]
        else:
            class_name = f"Unknown({class_id})"

        color = (0, 0, 255) if class_name == "broken" else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to display the image in the Tkinter window
def show_image(image_path):
    image = predict_image(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not process the image.")
        return

    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((600, 400), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image_pil)
    panel.config(image=image_tk)
    panel.image = image_tk

# Function to open a file dialog for image selection
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        show_image(file_path)

# Function to process video frames
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    def process_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
        output = results[0]

        for box, conf, cls in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)

            if class_id < len(classes):
                class_name = classes[class_id]
            else:
                class_name = f"Unknown({class_id})"

            color = (0, 0, 255) if class_name == "broken" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil)

        panel.config(image=frame_tk)
        panel.image = frame_tk

        panel.after(30, process_frame)

    process_frame()

# Function to open a file dialog for video selection
def upload_video():
    file_path = filedialog.askopenfilename(
        title="Select a Video",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov")]
    )
    if file_path:
        predict_video(file_path)

# Tkinter window setup
root = tk.Tk()
root.title("License Plate Damage Detection")
root.geometry("800x600")

panel = tk.Label(root)
panel.pack(padx=10, pady=10)

btn_image = tk.Button(root, text="Upload Image", command=upload_image)
btn_image.pack(side="left", padx=20, pady=20)

btn_video = tk.Button(root, text="Upload Video", command=upload_video)
btn_video.pack(side="right", padx=20, pady=20)

root.mainloop()
