try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "The 'ultralytics' package is not installed or could not be imported. "
        "Install it with: pip install ultralytics"
    ) from e

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')

# Train on your dataset
results = model.train(
    data='/Users/sudhir/Desktop/NJDOT/Model/US Road Signs.v1i.yolov8/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name="US_Road_Signs_Model",
    device='cpu'   #  train on CPU
)

# Save the best model
model.save("best_model.pt")

print(" Training complete! Model saved as best_model.pt")
