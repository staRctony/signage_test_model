from ultralytics import YOLO

# Load your trained YOLOv8 model
m = YOLO("/Users/sudhir/Desktop/NJDOT/Model/US Road Signs.v1i.yolov8/best_model.pt")

# Export to TFLite with specified parameters
m.export(
    format="tflite",
    imgsz=640,       
    nms=True,        # Enable Non-Maximum Suppression for [N,6] output (x1,y1,x2,y2,score,cls)
    dynamic=False,   # Fix input shape to [1,640,640,3]
    int8=False       # Use fp32 (default) for compatibility
)

# The TFLite model will be saved in the runs directory (e.g., runs/segment/train/weights/best_model.tflite)
print("Model successfully converted to TensorFlow Lite format.")