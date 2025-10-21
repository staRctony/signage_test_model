from ultralytics import YOLO
import tensorflow as tf

# Load your trained YOLOv8 model
model = YOLO("/Users/sudhir/Desktop/NJDOT/Model/US Road Signs.v1i.yolov8/best_model.pt")

# Export to TensorFlow SavedModel
model.export(format="tf")

# Load the SavedModel
saved_model_dir = "/Users/sudhir/Desktop/NJDOT/Model/US Road Signs.v1i.yolov8/best_model_saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Convert to TFLite
tflite_model = converter.convert()

# Save the TFLite model
with open("/Users/sudhir/Desktop/NJDOT/Model/US Road Signs.v1i.yolov8/best_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format.")