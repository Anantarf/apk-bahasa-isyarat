import cv2
import time

print("Detecting all available cameras...\n")

for i in range(3):
    print(f"Testing Camera Index {i}:")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [OK] Available - Resolution: {width}x{height}")
        
        # Try capture frame
        ret, frame = cap.read()
        if ret and frame is not None:
            actual_h, actual_w = frame.shape[:2]
            print(f"  Actual frame: {actual_w}x{actual_h}")
            
            if abs(actual_w - width) > 50 or abs(actual_h - height) > 50:
                print(f"  [WARN] SUSPICIOUS: Resolution mismatch (possible virtual camera)")
        else:
            print(f"  [WARN] Frame capture failed")
    else:
        print(f"  [X] Not available")
    
    cap.release()
    print()
    time.sleep(0.3)

print("Detection complete!")

