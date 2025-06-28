import cv2
from HandTracking.tracker import HandTracker
from canvas.drawing import VirtualCanvas

def tracker():
    """Test the hand tracker with camera input"""
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    canvas = VirtualCanvas(640,480)
    
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
            # print("Index finger tip: ", index_tip)

            # Get middle finger tip
            middle_tip = tracker.middle_finger_up(frame)
            # print("Middle finger tip: ", middle_tip)
            # if middle_tip:
            #     print("Middle finger tip detected")
            #     break
            canvas.update(index_tip)
            
            if index_tip:
                x, y = index_tip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"Tip: ({x}, {y})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                canvas.update((x, y))
                if middle_tip:
                    canvas.clear()
            else:
                cv2.putText(frame, "No index finger tip detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            canvas_bgr = cv2.cvtColor(canvas.get_image(), cv2.COLOR_GRAY2BGR)
            combined = cv2.addWeighted(frame, 1, canvas_bgr, 0.5, 0)
    
            cv2.imshow('Hand Tracker Test', frame)
            cv2.imshow('Canvas', canvas.get_image())
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker() 