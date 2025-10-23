import os
import time
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# === Config & Environment ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# === Initialize model ===
MODEL_PATH = "Q1.plate _recognition\\yolo11m.pt"  # change if needed
try:
    model = YOLO(MODEL_PATH)
except Exception:
    print("⚠️ Model not found — using untrained YOLO model.")
    model = YOLO()

classes = ['broken', 'non broken']
BROKEN_DIR = "broken_plates"
os.makedirs(BROKEN_DIR, exist_ok=True)

# === Globals ===
cap = None
running = False
video_running = False
last_saved_centers = []
VEHICLE_SAVE_THRESHOLD = 60

# === Utility Functions ===
def open_camera_auto(max_index=4):
    for i in range(max_index + 1):
        flag = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        c = cv2.VideoCapture(i, flag)
        if c is not None and c.isOpened():
            try:
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return c, i
        try:
            c.release()
        except Exception:
            pass
    return None, None

def is_new_center(cx, cy, threshold=VEHICLE_SAVE_THRESHOLD):
    global last_saved_centers
    for (pcx, pcy) in last_saved_centers:
        if abs(cx - pcx) < threshold and abs(cy - pcy) < threshold:
            return False
    return True

def get_next_vehicle_count():
    files = [f for f in os.listdir(BROKEN_DIR) if f.lower().startswith("vehicle_") and f.lower().endswith((".png", ".jpg", ".jpeg"))]
    max_n = 0
    for f in files:
        try:
            n = int(os.path.splitext(f)[0].split("_")[-1])
            if n > max_n:
                max_n = n
        except Exception:
            pass
    return max_n + 1

def extract_detections(output):
    dets = []
    try:
        boxes = getattr(output.boxes, "xyxy", None)
        confs = getattr(output.boxes, "conf", None)
        clss = getattr(output.boxes, "cls", None)
        if boxes is None or len(boxes) == 0:
            return dets
        boxes_arr = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.array(boxes)
        confs_arr = confs.cpu().numpy() if hasattr(confs, "cpu") else np.array(confs)
        clss_arr = clss.cpu().numpy() if hasattr(clss, "cpu") else np.array(clss)
        for i in range(len(boxes_arr)):
            x1, y1, x2, y2 = map(int, boxes_arr[i])
            conf = float(confs_arr[i]) if i < len(confs_arr) else 0.0
            cls = int(clss_arr[i]) if i < len(clss_arr) else 0
            dets.append((x1, y1, x2, y2, conf, cls))
    except Exception:
        try:
            for b, c, cl in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
                x1, y1, x2, y2 = map(int, b)
                conf = float(c)
                cls = int(cl)
                dets.append((x1, y1, x2, y2, conf, cls))
        except Exception:
            pass
    return dets

# === Prediction / Drawing ===
def predict_frame(frame):
    global last_saved_centers
    try:
        results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
    except Exception as e:
        print("Model prediction error:", e)
        return frame, "Model error", "gray", False

    if not results:
        return frame, "No plate detected", "gray", False

    output = results[0]
    dets = extract_detections(output)

    detected_label = "No plate detected"
    detected_color = "gray"
    broken_found = False
    normal_found = False

    for (x1, y1, x2, y2, conf, cls) in dets:
        class_id = int(cls)
        class_name = classes[class_id] if class_id < len(classes) else f"Unknown({class_id})"
        display_name = "Broken Plate" if class_name == "broken" else "Normal Plate"
        color_bgr = (0, 0, 255) if class_name == "broken" else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{display_name}: {conf:.2f}"
        cv2.putText(frame, label, (max(0, x1), max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

        if class_name == "broken":
            detected_label = "Broken Plate Detected"
            detected_color = "red"
            broken_found = True
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if is_new_center(cx, cy):
                h, w = frame.shape[:2]
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(w - 1, x2), min(h - 1, y2)
                crop = frame[ry1:ry2, rx1:rx2].copy()
                if crop.size != 0:
                    vnum = get_next_vehicle_count()
                    fname = f"vehicle_{vnum}.png"
                    path = os.path.join(BROKEN_DIR, fname)
                    try:
                        cv2.imwrite(path, crop)
                        last_saved_centers.append((cx, cy))
                    except Exception as e:
                        print("Failed to save vehicle crop:", e)
        else:
            normal_found = True

    if not broken_found and normal_found:
        detected_label = "No broken plates detected"
        detected_color = "blue"

    return frame, detected_label, detected_color, broken_found

# === Display Helpers ===
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

# === Camera & Video ===
def start_camera():
    global cap, running
    remove_uploaded_image()
    if running:
        return
    cap_found, idx = open_camera_auto(max_index=4)
    if cap_found is None:
        messagebox.showerror("Error", "Cannot access any camera.")
        return
    cap = cap_found
    running = True
    status_label.config(text=f"📷 Camera Started (index {idx})", fg="blue")
    btn_start_camera.config(state="disabled")
    btn_stop_camera.config(state="normal")
    panel.after(1, update_camera)

def stop_camera():
    global cap, running
    running = False
    if cap:
        try:
            cap.release()
        except Exception:
            pass
    status_label.config(text="🛑 Camera Stopped", fg="gray")
    btn_start_camera.config(state="normal")
    btn_stop_camera.config(state="disabled")

def update_camera():
    global cap, running
    if not running or cap is None:
        return
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 🔁 Mirror the frame horizontally

    if not ret or frame is None:
        stop_camera()
        return
    annotated, label, color, broken = predict_frame(frame.copy())
    update_display_bgr(annotated)
    status_label.config(text=label, fg=color)
    if running:
        panel.after(1, update_camera)

def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not file_path:
        return
    remove_uploaded_image()
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Could not read the image.")
        return
    annotated, label, color, broken = predict_frame(img.copy())
    update_display_bgr(annotated)
    status_label.config(text=f"✅ {label}", fg=color)

def play_video(path):
    global video_running
    remove_uploaded_image()
    cap_local = cv2.VideoCapture(path)
    if not cap_local.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return
    video_running = True
    def process():
        global video_running
        if not video_running:
            try:
                cap_local.release()
            except Exception:
                pass
            return
        ret, frame = cap_local.read()
        if not ret:
            cap_local.release()
            status_label.config(text="✅ Video Finished", fg="green")
            video_running = False
            return
        annotated, label, color, broken = predict_frame(frame.copy())
        update_display_bgr(annotated)
        status_label.config(text=label, fg=color)
        panel.after(1, process)
    status_label.config(text="🔄 Processing Video...", fg="blue")
    process()

def upload_video():
    file_path = filedialog.askopenfilename(title="Select a Video",
                                           filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        play_video(file_path)

def remove_uploaded_image():
    global cap, running, video_running
    panel.config(image="", text="")
    if hasattr(panel, "image"):
        delattr(panel, "image")
    if hasattr(panel, "image_array"):
        delattr(panel, "image_array")
    if running and cap:
        running = False
        try:
            cap.release()
        except Exception:
            pass
        btn_start_camera.config(state="normal")
        btn_stop_camera.config(state="disabled")
    video_running = False
    status_label.config(text="Preview cleared", fg="gray")

# === Broken Plate Viewer (Scrollable) ===
def view_saved_broken_plates():
    files = sorted([f for f in os.listdir(BROKEN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not files:
        messagebox.showinfo("Info", "No broken plates saved yet.")
        return
    win = tk.Toplevel(root)
    win.title("Saved Broken Plates")
    win.geometry("700x500")

    canvas_win = tk.Canvas(win, bg="#ecf0f1")
    v_scroll = tk.Scrollbar(win, orient="vertical", command=canvas_win.yview)
    canvas_win.configure(yscrollcommand=v_scroll.set)
    v_scroll.pack(side="right", fill="y")
    canvas_win.pack(side="left", fill="both", expand=True)

    frame_win = tk.Frame(canvas_win, bg="#ecf0f1")
    canvas_win.create_window((0, 0), window=frame_win, anchor="nw")

    for f in files:
        path = os.path.join(BROKEN_DIR, f)
        try:
            img_cv = cv2.imread(path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_cv)
            img_pil.thumbnail((600, 320))
            imgtk = ImageTk.PhotoImage(img_pil)
            sub = tk.Frame(frame_win, bg="#ffffff", bd=1, relief="solid")
            lbl = tk.Label(sub, image=imgtk)
            lbl.image = imgtk
            lbl.pack(padx=6, pady=6)
            tk.Label(sub, text=f, bg="#ffffff").pack(pady=(0, 6))
            sub.pack(fill="x", padx=8, pady=8)
        except Exception as e:
            print("Failed to load:", e)

    def on_mousewheel(event):
        canvas_win.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas_win.bind_all("<MouseWheel>", on_mousewheel)
    frame_win.update_idletasks()
    canvas_win.config(scrollregion=canvas_win.bbox("all"))

# === Exit Cleanup ===
def exit_app():
    global cap, running, video_running
    running = False
    video_running = False
    if cap and cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
    root.destroy()

# === GUI Layout ===
root = tk.Tk()
root.title("License Plate Damage Detection")
root.geometry("980x900")
root.minsize(900, 700)
root.configure(bg="#ecf0f1")

header = tk.Frame(root, bg="#2c3e50", height=70)
header.pack(fill="x")
tk.Label(header, text="License Plate Damage Detection", font=("Segoe UI", 22, "bold"), fg="white", bg="#2c3e50").pack(pady=10)

main_frame = tk.Frame(root, bg="#ecf0f1")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

status_label = tk.Label(main_frame, text="Upload an image, video, or start the camera",
                        font=("Segoe UI", 14, "bold"), bg="#ecf0f1", fg="gray")
status_label.pack(pady=8)

panel_frame = tk.Frame(main_frame, bg="#bdc3c7")
panel_frame.pack(fill="both", expand=True, pady=12)
panel_frame.pack_propagate(False)
panel = tk.Label(panel_frame, bg="white", relief="sunken")
panel.pack(fill="both", expand=True)

btn_frame = tk.Frame(main_frame, bg="#ecf0f1")
btn_frame.pack(fill="x", pady=16)
btn_frame.grid_columnconfigure((0, 1, 2), weight=1)

btn_style = {"font": ("Segoe UI", 12, "bold"), "height": 2, "relief": "raised", "bd": 2}

tk.Button(btn_frame, text="🖼 Upload Image", command=upload_image, bg="#3498db", fg="white",
          **btn_style).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
tk.Button(btn_frame, text="🗑 Remove Image/Video", command=remove_uploaded_image,
          bg="#95a5a6", fg="white", **btn_style).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
tk.Button(btn_frame, text="🎥 Upload Video", command=upload_video,
          bg="#27ae60", fg="white", **btn_style).grid(row=0, column=2, sticky="ew", padx=8, pady=6)

btn_start_camera = tk.Button(btn_frame, text="📸 Start Camera", command=start_camera,
                             bg="#f1c40f", fg="black", **btn_style)
btn_start_camera.grid(row=1, column=0, sticky="ew", padx=8, pady=6)

btn_stop_camera = tk.Button(btn_frame, text="🛑 Stop Camera", command=stop_camera,
                            bg="#e67e22", fg="white", **btn_style, state="disabled")
btn_stop_camera.grid(row=1, column=1, sticky="ew", padx=8, pady=6)

tk.Button(btn_frame, text="🖼 View Saved Broken Plates", command=view_saved_broken_plates,
          bg="#9b59b6", fg="white", **btn_style).grid(row=2, column=0, sticky="ew", padx=8, pady=6)

tk.Button(btn_frame, text="Exit", command=exit_app, bg="#e74c3c",
          fg="white", font=("Segoe UI", 12, "bold"), width=18).grid(row=1, column=2, sticky="ew", padx=8, pady=6)

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

root.protocol("WM_DELETE_WINDOW", exit_app)
root.mainloop()
