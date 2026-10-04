# Webapp model slot

Drop the trained YOLOv8 ONNX file here as:

- `yolov8n-roadguard.onnx` (nominal name used by `config.js`)

The webapp HEAD-checks the file at load time and quietly falls back to a
mock detector when it is missing, so the UI shell is testable without a
trained model.

Export from the notebook with:

```python
from ultralytics import YOLO
model = YOLO("runs/detect/roadguard-yolov8n/weights/best.pt")
model.export(format="onnx", opset=12, dynamic=False, imgsz=640)
```

Then copy the resulting `.onnx` into this directory. The file is
gitignored.
