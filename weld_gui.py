import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter

MODEL_PATH = "runs/obb/train9/weights/best.pt"
RESULTS_DIR = "results"

class WeldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Welding Defect Detection (YOLOv8 OBB)")
        self.geometry("1100x600")
        self.minsize(900, 500)
        self.configure(bg="#f0f0f0")
        self.center_window()
        # Load multiple models for ensemble if available
        self.models = []
        self.models.append(YOLO(MODEL_PATH))
        
        # Try to load other trained models for ensemble
        for train_dir in ['train8', 'train7', 'train6']:
            try:
                alt_path = f"runs/obb/{train_dir}/weights/best.pt"
                if os.path.exists(alt_path):
                    self.models.append(YOLO(alt_path))
                    print(f"Loaded ensemble model: {alt_path}")
            except:
                pass
        
        for model in self.models:
            model.to('cpu')
        print(f"Loaded {len(self.models)} models for ensemble prediction")
        self.file_path = None
        self.result_img = None
        self.orig_img = None
        self.is_video = False
        self.create_widgets()
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

    def center_window(self):
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        ws = self.winfo_screenwidth()
        hs = self.winfo_screenheight()
        x = (ws // 2) - (w // 2)
        y = (hs // 2) - (h // 2)
        self.geometry(f"+{x}+{y}")

    def create_widgets(self):
        top_frame = tk.Frame(self, bg="#f0f0f0")
        top_frame.pack(side=tk.TOP, pady=10)

        btn_select = tk.Button(top_frame, text="Select Image/Video", command=self.select_file, width=18, font=("Arial", 12))
        btn_select.pack(side=tk.LEFT, padx=10)

        btn_run = tk.Button(top_frame, text="Run Detection", command=self.run_detection, width=15, font=("Arial", 12))
        btn_run.pack(side=tk.LEFT, padx=10)

        btn_save = tk.Button(top_frame, text="Save Result", command=self.save_result, width=13, font=("Arial", 12))
        btn_save.pack(side=tk.LEFT, padx=10)

        main_frame = tk.Frame(self, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.canvas_orig = tk.Canvas(main_frame, bg="#d9d9d9", width=500, height=400)
        self.canvas_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.canvas_orig.create_text(250, 200, text="Original", font=("Arial", 16), fill="gray")

        self.canvas_result = tk.Canvas(main_frame, bg="#d9d9d9", width=500, height=400)
        self.canvas_result.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        self.canvas_result.create_text(250, 200, text="Detection Result", font=("Arial", 16), fill="gray")

        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        self.show_images()

    def select_file(self):
        filetypes = [("Image/Video Files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov")]
        path = filedialog.askopenfilename(title="Select Image or Video", filetypes=filetypes)
        if path:
            self.file_path = path
            self.is_video = path.lower().endswith(('.mp4', '.avi', '.mov'))
            self.load_original()
            self.result_img = None
            self.show_images()

    def load_original(self):
        if not self.file_path:
            return
        if self.is_video:
            cap = cv2.VideoCapture(self.file_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self.orig_img = None
        else:
            img = cv2.imread(self.file_path)
            if img is not None:
                self.orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                self.orig_img = None

    def advanced_detection(self, input_img):
        """Simplified but accurate detection with ensemble and multi-scale"""
        all_detections = []
        
        # Multi-scale detection only (remove TTA transforms that cause issues)
        scales = [0.8, 1.0, 1.2]
        
        for model in self.models:
            for scale in scales:
                try:
                    # Resize image for multi-scale detection
                    h, w = input_img.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_img = cv2.resize(input_img, (new_w, new_h))
                    
                    # Multiple confidence thresholds for robust detection
                    for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        results = model.predict(
                            scaled_img,
                            conf=conf,
                            iou=0.3,
                            device='cpu',
                            verbose=False,
                            save=False,
                            show=False,
                            agnostic_nms=True,
                            max_det=100
                        )
                        
                        if results and len(results) > 0:
                            result = results[0]
                            
                            # Extract detections
                            if hasattr(result, 'obb') and result.obb is not None:
                                if hasattr(result.obb, 'data') and result.obb.data is not None and len(result.obb.data) > 0:
                                    detections = result.obb.data.cpu().numpy()
                                    for det in detections:
                                        # Scale back coordinates if needed
                                        scaled_det = det.copy()
                                        if scale != 1.0:
                                            # Scale back OBB coordinates (8 points)
                                            if len(scaled_det) >= 8:
                                                scaled_det[:8] /= scale
                                        all_detections.append((scaled_det, conf, model))
                            
                            elif hasattr(result, 'boxes') and result.boxes is not None:
                                if hasattr(result.boxes, 'data') and result.boxes.data is not None and len(result.boxes.data) > 0:
                                    detections = result.boxes.data.cpu().numpy()
                                    for det in detections:
                                        scaled_det = det.copy()
                                        if scale != 1.0:
                                            # Scale back box coordinates (4 points)
                                            scaled_det[:4] /= scale
                                        all_detections.append((scaled_det, conf, model))
                except Exception as e:
                    print(f"Detection error at scale {scale}: {e}")
                    continue
        
        return self.filter_detections(all_detections, input_img)

    def filter_detections(self, all_detections, input_img):
        """Intelligent filtering with confidence-based selection"""
        if not all_detections:
            return None
        
        # Sort by confidence (highest first)
        all_detections.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only high-confidence detections
        high_conf_detections = [det for det, conf, _ in all_detections if conf >= 0.3]
        
        if not high_conf_detections:
            # If no high confidence, take medium confidence
            medium_conf_detections = [det for det, conf, _ in all_detections if conf >= 0.2]
            return medium_conf_detections[:3] if medium_conf_detections else None
        
        # Apply spatial filtering to remove duplicates
        filtered_detections = []
        for det, conf, model in all_detections:
            if conf < 0.25:
                continue
                
            # Check for spatial overlap with existing detections
            is_duplicate = False
            for existing_det, _, _ in filtered_detections:
                if self.calculate_overlap(det, existing_det) > 0.6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_detections.append((det, conf, model))
        
        return filtered_detections[:5]  # Return top 5 detections

    def calculate_overlap(self, det1, det2):
        """Calculate overlap between two detections"""
        try:
            # Simple distance-based overlap for now
            if len(det1) >= 4 and len(det2) >= 4:
                center1 = (det1[0] + det1[2]) / 2, (det1[1] + det1[3]) / 2
                center2 = (det2[0] + det2[2]) / 2, (det2[1] + det2[3]) / 2
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Normalize by image size
                img_diag = np.sqrt(640**2 + 640**2)
                normalized_distance = distance / img_diag
                
                return 1.0 - min(normalized_distance, 1.0)
            return 0.0
        except:
            return 0.0

    def run_detection(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Please select an image or video first.")
            return
        
        self.result_message = None
        print(f"Starting detection on: {self.file_path}")
        
        try:
            # Load image
            if self.is_video:
                cap = cv2.VideoCapture(self.file_path)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    self.result_img = None
                    self.result_message = "Could not read video file."
                    self.show_images()
                    return
                input_img = frame
            else:
                input_img = cv2.imread(self.file_path)
                if input_img is None:
                    self.result_img = None
                    self.result_message = "Could not read image file."
                    self.show_images()
                    return
            
            print(f"Image shape: {input_img.shape}")
            
            # Try ensemble detection first
            detections = self.advanced_detection(input_img)
            
            if detections and len(detections) > 0:
                print(f"Ensemble found {len(detections)} detections")
                # Use best model for visualization
                best_model = self.models[0]
                results = best_model.predict(
                    input_img,
                    conf=0.2,
                    iou=0.4,
                    device='cpu',
                    verbose=False,
                    save=False,
                    show=False
                )
                
                if results and len(results) > 0:
                    img_out = results[0].plot()
                    self.result_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
                    self.result_message = f"Found {len(detections)} defects"
                else:
                    self.result_img = self.orig_img.copy()
                    self.result_message = f"Found {len(detections)} defects (plot failed)"
            else:
                # Single model fallback with very low confidence
                print("Trying single model with low confidence...")
                best_model = self.models[0]
                results = best_model.predict(
                    input_img,
                    conf=0.05,
                    iou=0.3,
                    device='cpu',
                    verbose=True,
                    save=False,
                    show=False
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    detection_count = 0
                    
                    if hasattr(result, 'obb') and result.obb is not None:
                        if hasattr(result.obb, 'data') and result.obb.data is not None:
                            detection_count = len(result.obb.data)
                    elif hasattr(result, 'boxes') and result.boxes is not None:
                        if hasattr(result.boxes, 'data') and result.boxes.data is not None:
                            detection_count = len(result.boxes.data)
                    
                    if detection_count > 0:
                        img_out = result.plot()
                        self.result_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
                        self.result_message = f"Found {detection_count} low-confidence defects"
                    else:
                        self.result_img = None
                        self.result_message = "No defects detected"
                else:
                    self.result_img = None
                    self.result_message = "No defects detected"
            
            self.show_images()
            
        except Exception as e:
            print(f"Detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.result_img = None
            self.result_message = f"Detection failed: {str(e)}"
            self.show_images()
            messagebox.showerror("Detection Error", str(e))

    def show_images(self):
        # Display original
        self.display_on_canvas(self.canvas_orig, self.orig_img)
        # Display result
        self.display_on_canvas(self.canvas_result, self.result_img, self.result_message if hasattr(self, 'result_message') else None)

    def display_on_canvas(self, canvas, img_arr, message=None):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if img_arr is not None:
            img = Image.fromarray(img_arr)
            img.thumbnail((w, h), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            canvas.create_image(w//2, h//2, image=self.tk_img, anchor=tk.CENTER)
        else:
            txt = message if message else ("Original" if canvas == self.canvas_orig else "Detection Result")
            canvas.create_text(w//2, h//2, text=txt, font=("Arial", 16), fill="gray")

    def save_result(self):
        if self.result_img is None:
            messagebox.showwarning("No result", "No detection result to save.")
            return
        base = os.path.basename(self.file_path) if self.file_path else "result.png"
        name, ext = os.path.splitext(base)
        save_path = os.path.join(RESULTS_DIR, f"{name}_detected.png")
        img = Image.fromarray(self.result_img)
        img.save(save_path)
        messagebox.showinfo("Saved", f"Result saved to {save_path}")

if __name__ == "__main__":
    app = WeldApp()
    app.mainloop()
