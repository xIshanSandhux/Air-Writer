import cv2
from HandTracking.tracker import HandTracker

def tracker():
    """Test the hand tracker with camera input"""
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Testing hand tracker...")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Get index finger tip
            index_tip = tracker.get_index_finger_tip(frame)
            print("Index finger tip: ", index_tip)

            # Get middle finger tip
            middle_tip = tracker.middle_finger_up(frame)
            print("Middle finger tip: ", middle_tip)
            if middle_tip:
                print("Middle finger tip detected")
                break
            
            
            if index_tip:
                x, y = index_tip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"Tip: ({x}, {y})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No index finger tip detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Hand Tracker Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker() 