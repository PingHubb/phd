import cv2
from ultralytics import YOLO

DEV = "/dev/video4"

def main():
    # 1. Load the model.
    # 'yolov8n.pt' is the "Nano" version. It is the fastest and best for robots/CPUs.
    # It will automatically download the file the first time you run the script.
    print("Loading AI Model...")
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(DEV)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {DEV}. Close Cheese and try again.")

    print(f"Camera opened on {DEV}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # 2. Feed the frame to the model to detect objects
        # stream=True makes it run faster for video
        results = model(frame, stream=True, verbose=False)

        # 3. Process the results
        # The model returns a list of results. We only have one frame, so we iterate.
        for result in results:
            # This function draws the boxes and labels directly onto the frame for you
            annotated_frame = result.plot()

        # 4. Show the frame with the boxes (annotated_frame), not the raw frame
        cv2.imshow(f"Live Object Detection: {DEV}", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()