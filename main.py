import cv2
from HandTracking.tracker import HandTracker
from canvas.drawing import VirtualCanvas


tracker = HandTracker()
canvas = VirtualCanvas()
text = ""

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if ret:
        frame = cv2.flip(frame, 1)

    point = tracker.get_index_finger_tip(frame)
    if point:
        x, y = point
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        canvas.update(point)
        # cv2.putText(frame, f"Tip: ({x}, {y})", (10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # if tracker.get_thumb_tip(frame):
    #     print("Thumb tip detected")
    #     canvas.clear()       
        


   

    # Overlay canvas on frame
    canvas_img = canvas.get_image()
    combined = cv2.addWeighted(frame, 0.5, cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR), 0.5, 0)

    cv2.imshow("AirWriter", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # print("Typed text:", text)

cap.release()
cv2.destroyAllWindows()
