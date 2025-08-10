# Welding AI Demo Project

## Project Overview
This project demonstrates computer vision capabilities for welding applications using YOLOv8 Oriented Object Detection. The dataset was created and exported from Roboflow for training and deploying AI models to detect and analyze welding-related objects.

## Dataset Information
- **Project Name**: My First Project - v1
- **Export Date**: August 9, 2025 at 5:00 AM GMT
- **Dataset Size**: 19 images
- **Annotation Format**: YOLOv8 Oriented Object Detection
- **Source**: [Roboflow Universe](https://universe.roboflow.com/weld-r7cc6/my-first-project-ny8uf)
- **License**: CC BY 4.0

## Features
- Object detection and recognition in welding environments
- Oriented bounding box detection for precise object localization
- Pre-processed dataset ready for training
- Compatible with YOLOv8 architecture

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 8GB RAM
- 5GB+ free disk space

### Dependencies
```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.3.0
pillow>=8.0.0
torch>=1.9.0
torchvision>=0.10.0
roboflow>=1.0.0
```

### Installation
```bash
pip install ultralytics roboflow opencv-python numpy matplotlib pillow
```

## Workflow

### 1. Data Preparation
- Dataset exported from Roboflow with 19 annotated images
- No image augmentation applied during preprocessing
- Objects annotated in YOLOv8 Oriented Object Detection format

### 2. Model Training
```bash
# Train YOLOv8 model
yolo train data=path/to/dataset.yaml model=yolov8n-obb.pt epochs=100 imgsz=640
```

### 3. Model Validation
```bash
# Validate trained model
yolo val model=runs/train/exp/weights/best.pt data=path/to/dataset.yaml
```

### 4. Inference
```bash
# Run inference on new images
yolo predict model=runs/train/exp/weights/best.pt source=path/to/images
```

### 5. Deployment
- Export model to desired format (ONNX, TensorRT, etc.)
- Integrate with production pipeline
- Monitor performance and collect feedback

## Project Structure
```
demoai/
├── README.md
├── README.roboflow.txt
├── README.dataset.txt
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
├── results/
└── scripts/
```

## Usage Examples

### Basic Training
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-obb.pt')

# Train the model
results = model.train(data='dataset.yaml', epochs=100, imgsz=640)
```

### Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('path/to/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results.show()
```

## Performance Metrics
- Mean Average Precision (mAP)
- Precision and Recall
- Inference Speed (FPS)
- Model Size

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Resources
- [Roboflow Platform](https://roboflow.com)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [Computer Vision Notebooks](https://github.com/roboflow/notebooks)
- [Roboflow Universe](https://universe.roboflow.com)

## License
This project is licensed under CC BY 4.0 - see the dataset source for details.

## Contact
For questions or support, please refer to the Roboflow community or project maintainers.
