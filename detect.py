from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load multiple models for ensemble
models = []
model_paths = [
    "runs/obb/train9/weights/best.pt",
    "runs/obb/train8/weights/best.pt", 
    "runs/obb/train7/weights/best.pt"
]

for path in model_paths:
    if os.path.exists(path):
        try:
            model = YOLO(path)
            model.to('cpu')
            models.append(model)
            print(f"Loaded: {path}")
        except:
            print(f"Failed to load: {path}")

if not models:
    print("No models loaded!")
    exit(1)

# Test on an image
img_path = "test_weld.jpg"
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    exit(1)

img = cv2.imread(img_path)
if img is None:
    print("Failed to load image.")
    exit(1)

print(f"Image loaded: {img.shape}")
print(f"Using {len(models)} models for ensemble detection")

# Simplified ensemble detection without augmentation issues
all_results = []
confidence_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

for i, model in enumerate(models):
    print(f"\n=== Model {i+1} Results ===")
    for conf in confidence_levels:
        try:
            results = model.predict(
                img,
                conf=conf,
                iou=0.4,
                device='cpu',
                verbose=False,
                save=False,
                show=False,
                agnostic_nms=True,
                max_det=50
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
                    print(f"Model {i+1} found {detection_count} detections at conf={conf}")
                    all_results.append((results, conf, i+1, detection_count))
                    
                    # Show first detection found
                    if len(all_results) == 1:
                        img_out = result.plot()
                        cv2.imshow(f"First Detection (Model {i+1}, conf={conf})", img_out)
                        cv2.waitKey(2000)  # Show for 2 seconds
                        cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error with Model {i+1} at conf={conf}: {e}")
            continue

# Show best results
if all_results:
    # Sort by detection count and confidence
    all_results.sort(key=lambda x: (x[3], x[1]), reverse=True)
    
    print(f"\n=== Best Results ===")
    best_result, best_conf, best_model, best_count = all_results[0]
    
    print(f"Best: Model {best_model} with {best_count} detections at conf={best_conf}")
    
    img_out = best_result[0].plot()
    cv2.imshow(f"Best Detections (Model {best_model}, conf={best_conf})", img_out)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite("detection_result.jpg", img_out)
    print("Result saved as 'detection_result.jpg'")
else:
    print("No detections found across all models and confidence levels")
    print("This might indicate:")
    print("1. No defects in the image")
    print("2. Model needs retraining with more data")
    print("3. Image quality or format issues")
