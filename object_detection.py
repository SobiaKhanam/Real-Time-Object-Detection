
"""
Real-Time Object Detection on NVIDIA Jetson Nano

Uses a pretrained YOLO model to detect objects in a live
camera feed and display class labels, bounding boxes,
and confidence percentages.
"""

import cv2
from ultralytics import YOLO


# Configuration
MODEL_PATH = "yolov5s.pt"  # Pretrained YOLOv5 small model
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.50


def main():
    # Load pretrained YOLO weights.
    # The model file is downloaded automatically if supported
    # by the installed Ultralytics version and internet access.
    print("Loading pretrained YOLO model...")
    model = YOLO(MODEL_PATH)

    # Open the connected camera.
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        print("Check the camera connection and CAMERA_INDEX.")
        return

    print("Object detection started. Press 'q' to quit.")

    try:
        while True:
            # Read a frame from the camera.
            success, frame = camera.read()

            if not success:
                print("Warning: Unable to read camera frame.")
                break

            # Run object detection on the current frame.
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            # Draw detection boxes, class labels, and confidence.
            annotated_frame = results[0].plot()

            # Print each detected object's label and confidence.
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item()) * 100
                label = model.names[class_id]

                print(
                    f"Object: {label} | "
                    f"Confidence: {confidence:.1f}%"
                )

            # Show the annotated live camera feed.
            cv2.imshow(
                "Real-Time YOLO Object Detection",
                annotated_frame
            )

            # Exit when the user presses 'q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDetection stopped by user.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Application closed.")


if __name__ == "__main__":
    main()
