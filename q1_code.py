import os
import time
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# === Config & Environment ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# === Initialize model ===
try:
    model = YOLO("runs/detect/train/weights/best.pt")
except Exception:
    print("⚠️ best.pt not found — using untrained YOLO model.")
    model = YOLO()

classes = ['broken', 'non broken']
BROKEN_DIR = "broken_plates"
os.makedirs(BROKEN_DIR, exist_ok=True)

# === Prediction / drawing ===
def predict_frame(frame):
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
    output = results[0]

    detected_label = "No plate detected"
    detected_color = "gray"
    broken_found = False

    for box, conf, cls in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        class_name = classes[class_id] if class_id < len(classes) else f"Unknown({class_id})"

        display_name = "Broken Plate" if class_name == "broken" else "Normal Plate"
        color_bgr = (0, 0, 255) if class_name == "broken" else (0, 255, 0)

        if class_name == "broken":
            detected_label = "Broken Plate Detected"
            detected_color = "red"
            broken_found = True
        elif class_name == "non broken":
            if detected_label != "Broken Plate Detected":
                detected_label = "Normal Plate Detected"
                detected_color = "green"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{display_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    return frame, detected_label, detected_color, broken_found

# === GUI helper: update panel display ===
def update_display_bgr(frame_bgr, fit_to_panel=True):
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return
    img = Image.fromarray(frame_rgb)
    if fit_to_panel:
        w = max(400, panel.winfo_width() or 640)
        h = max(300, panel.winfo_height() or 420)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(img)
    panel.config(image=imgtk)
    panel.image = imgtk
    panel.image_array = frame_rgb

# === Save broken plate frame and refresh GUI ===
def save_broken_frame(frame):
    ts = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    fname = f"broken_{ts}_{millis:03d}.png"
    path = os.path.join(BROKEN_DIR, fname)
    try:
        cv2.imwrite(path, frame)
        refresh_broken_plates()  # update GUI immediately
        return path
    except Exception as e:
        print("Failed to save broken frame:", e)
        return None

# === Image upload ===
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Could not read the image.")
        return
    annotated, label, color, broken = predict_frame(img.copy())
    update_display_bgr(annotated)
    status_label.config(text=f"✅ {label}", fg=color)
    if broken:
        saved = save_broken_frame(annotated)
        if saved:
            print("Saved broken plate image:", saved)

# === Remove / Clear uploaded image or panel content ===
def remove_uploaded_image():
    panel.config(image="", text="")
    if hasattr(panel, "image"):
        delattr(panel, "image")
    if hasattr(panel, "image_array"):
        delattr(panel, "image_array")
    status_label.config(text="Preview cleared", fg="gray")

# === Video upload and play ===
def play_video(path):
    cap_local = cv2.VideoCapture(path)
    if not cap_local.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    def process():
        ret, frame = cap_local.read()
        if not ret:
            cap_local.release()
            status_label.config(text="✅ Video Finished", fg="green")
            return
        annotated, label, color, broken = predict_frame(frame.copy())
        update_display_bgr(annotated)
        status_label.config(text=label, fg=color)
        if broken:
            save_broken_frame(annotated)
        panel.after(30, process)

    status_label.config(text="🔄 Processing Video...", fg="blue")
    process()

def upload_video():
    file_path = filedialog.askopenfilename(title="Select a Video",
                                           filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        play_video(file_path)

# === Camera / live feed ===
cap = None
running = False

def start_camera():
    global cap, running
    if running:
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera.")
        return
    running = True
    status_label.config(text="📷 Camera Started", fg="blue")
    btn_start_camera.config(state="disabled")
    btn_stop_camera.config(state="normal")
    update_camera()

def stop_camera():
    global cap, running
    running = False
    if cap and cap.isOpened():
        cap.release()
    cap = None
    status_label.config(text="🛑 Camera Stopped", fg="gray")
    btn_start_camera.config(state="normal")
    btn_stop_camera.config(state="disabled")

def update_camera():
    global cap, running
    if not running or cap is None:
        return
    ret, frame = cap.read()
    if not ret:
        stop_camera()
        return
    annotated, label, color, broken = predict_frame(frame.copy())
    update_display_bgr(annotated)
    status_label.config(text=label, fg=color)
    if broken:
        save_broken_frame(annotated)
    panel.after(20, update_camera)

# === Main GUI ===
root = tk.Tk()
root.title("License Plate Damage Detection")
root.geometry("980x820")
root.minsize(900, 700)
root.configure(bg="#ecf0f1")

# Header
header = tk.Frame(root, bg="#2c3e50", height=70)
header.pack(fill="x")
tk.Label(header, text="License Plate Damage Detection", font=("Segoe UI", 22, "bold"), fg="white", bg="#2c3e50").pack(pady=10)

# Main frame
main_frame = tk.Frame(root, bg="#ecf0f1")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Status label
status_label = tk.Label(main_frame, text="Upload an image, video, or start the camera", font=("Segoe UI", 14, "bold"), bg="#ecf0f1", fg="gray")
status_label.pack(pady=8)

# Panel frame for preview
panel_frame = tk.Frame(main_frame, bg="#bdc3c7", width=800, height=460)
panel_frame.pack(fill="both", expand=True, pady=12)
panel_frame.pack_propagate(False)
panel = tk.Label(panel_frame, bg="white", relief="sunken")
panel.pack(fill="both", expand=True)

# Buttons area
btn_frame = tk.Frame(main_frame, bg="#ecf0f1")
btn_frame.pack(pady=16)

btn_style = {"font": ("Segoe UI", 12, "bold"), "width": 18, "height": 2, "relief": "raised", "bd": 2}

tk.Button(btn_frame, text="🖼 Upload Image", command=upload_image, bg="#3498db", fg="white", **btn_style).grid(row=0, column=0, padx=8, pady=6)
tk.Button(btn_frame, text="🗑 Remove Image", command=remove_uploaded_image, bg="#95a5a6", fg="white", **btn_style).grid(row=0, column=1, padx=8, pady=6)
tk.Button(btn_frame, text="🎥 Upload Video", command=upload_video, bg="#27ae60", fg="white", **btn_style).grid(row=0, column=2, padx=8, pady=6)
tk.Button(btn_frame, text="📸 Start Camera", command=start_camera, bg="#f1c40f", fg="black", **btn_style).grid(row=1, column=0, padx=8, pady=6)
tk.Button(btn_frame, text="🛑 Stop Camera", command=stop_camera, bg="#e67e22", fg="white", **btn_style, state="disabled").grid(row=1, column=1, padx=8, pady=6)
tk.Button(btn_frame, text="Exit", command=lambda: (stop_camera(), root.destroy()), bg="#e74c3c", fg="white", font=("Segoe UI", 12, "bold"), width=18).grid(row=1, column=2, padx=8, pady=6)

# === Scrollable horizontal frame for broken plates ===
broken_frame_outer = tk.Frame(main_frame, bg="#ecf0f1")
broken_frame_outer.pack(fill="x", pady=10)

canvas = tk.Canvas(broken_frame_outer, height=160, bg="#ecf0f1", highlightthickness=0)
h_scroll = tk.Scrollbar(broken_frame_outer, orient="horizontal", command=canvas.xview)
canvas.configure(xscrollcommand=h_scroll.set)
h_scroll.pack(side="bottom", fill="x")
canvas.pack(side="left", fill="x", expand=True)

broken_frame = tk.Frame(canvas, bg="#ecf0f1")
canvas.create_window((0,0), window=broken_frame, anchor="nw")

def refresh_broken_plates():
    for widget in broken_frame.winfo_children():
        widget.destroy()

    files = sorted([f for f in os.listdir(BROKEN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not files:
        tk.Label(broken_frame, text="No broken plates saved yet", bg="#ecf0f1", fg="gray").pack()
        return

    for idx, f in enumerate(files):
        path = os.path.join(BROKEN_DIR, f)
        try:
            img_cv = cv2.imread(path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_cv)
            img_pil.thumbnail((140, 120))
            imgtk = ImageTk.PhotoImage(img_pil)

            lbl = tk.Label(broken_frame, image=imgtk, bd=2, relief="raised")
            lbl.image = imgtk
            lbl.pack(side="left", padx=6, pady=6)

            # Right-click to delete
            def delete_file(event, filepath=path, widget=lbl):
                try:
                    os.remove(filepath)
                    widget.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Cannot delete: {e}")

            lbl.bind("<Button-3>", delete_file)  # Right click
        except Exception as e:
            print("Failed to load broken plate:", e)

    broken_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

# Initial load
refresh_broken_plates()

# Panel resize handling
def on_panel_resize(event=None):
    if hasattr(panel, "image_array"):
        img = Image.fromarray(panel.image_array)
        w = max(400, panel.winfo_width())
        h = max(300, panel.winfo_height())
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        panel_imgtk = ImageTk.PhotoImage(img)
        panel.config(image=panel_imgtk)
        panel.image = panel_imgtk

panel.bind("<Configure>", on_panel_resize)

root.mainloop()
