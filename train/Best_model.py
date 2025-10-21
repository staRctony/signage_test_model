from ultralytics import YOLO
import cv2

# ---------------- CONFIG ----------------
MODEL_PATH = '/Users/sudhir/Desktop/NJDOT/Model/US Road Signs.v1i.yolov8/best_model.pt'
VIDEO_PATH = '/Users/sudhir/Downloads/IMG_9173.MOV'
SAVE_OUTPUT = True
OUTPUT_PATH = 'output_annotated_highway.mp4'
CONF_THRESHOLD = 0.25
DEVICE = 'cpu'  # change to 'cuda' if you have GPU
# ---------------------------------------

# Load the trained YOLO model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Get video properties for saving
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer to save annotated video
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("▶️ Starting live YOLOv8 detection. Press 'q' to quit.")

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete!")
        break

    # Run YOLO inference on the frame
    results = model.predict(frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    
    # Annotate frame
    annotated_frame = results[0].plot()

    # Show annotated frame live
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Save annotated frame if enabled
    if SAVE_OUTPUT:
        out.write(annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ User stopped the video.")
        break

# Release resources
cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
