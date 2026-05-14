import cv2
from ultralytics import YOLO

def start_room_detection():
    # 1. Load the pre-trained YOLOv8 model (small version for speed)
    # This will automatically download the 'yolov8n.pt' file on first run
    model = YOLO('yolov8n.pt')

    # 2. Open the laptop camera (0 is usually the default internal webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Detection started. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Run detection on the current frame
        # 'conf=0.5' means it only shows objects it's at least 50% sure about
        results = model(frame, conf=0.5, verbose=False)

        # 4. Visualize the results on the frame
        # 'plot()' draws the bounding boxes and labels for us
        annotated_frame = results[0].plot()

        # 5. Display the resulting frame
        cv2.imshow('Room Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_room_detection()