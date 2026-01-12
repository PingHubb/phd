import glob
import cv2

def try_device(dev_path: str) -> bool:
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return False

    # Try grabbing a few frames (some devices need a moment)
    ok = False
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            print(f"[OK] {dev_path} -> frame {w}x{h}")
            ok = True
            break

    cap.release()
    return ok

def main():
    devs = sorted(glob.glob("/dev/video*"))
    print("Found:", devs)
    for d in devs:
        print(f"Testing {d} ...", end=" ")
        if not try_device(d):
            print("NO")

if __name__ == "__main__":
    main()
