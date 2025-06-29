import cv2
from HandTracking.tracker import HandTracker
from canvas.drawing import VirtualCanvas
from tensorflow.keras.models import load_model
import numpy as np
from data.mapping import get_emnist_mapping
import time


label_map = get_emnist_mapping()
model = load_model("saved_models/emnist_cnn_model.h5")
def tracker():
    """Test the hand tracker with camera input"""
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    canvas = VirtualCanvas(640,480)
    text_output = ""
    
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
                    canvas_img = canvas.get_image()  # Your method to get the OpenCV canvas
                    preprocessed = preprocess_canvas(canvas_img)
                    letter = predict_character(preprocessed)
                    text_output += letter
                    print(f" Predicted letter: {letter}")
                    print(f" Text output: {text_output}")

                    canvas.clear()
                    time.sleep(1)
            else:
                cv2.putText(frame, "No index finger tip detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            canvas_bgr = cv2.cvtColor(canvas.get_image(), cv2.COLOR_GRAY2BGR)
            combined = cv2.addWeighted(frame, 1, canvas_bgr, 0.5, 0)
            cv2.putText(frame, f"Output: {text_output}", (10, 450), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
            cv2.imshow('Hand Tracker Test', frame)
            cv2.imshow('Canvas', canvas.get_image())
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def preprocess_canvas(canvas_img):
    # Convert to grayscale if needed
    if len(canvas_img.shape) == 3:
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)

    # Invert colors: EMNIST is black bg, white fg
    canvas_img = cv2.bitwise_not(canvas_img)

    # Resize and transpose to match EMNIST format
    canvas_img = cv2.resize(canvas_img, (28, 28))
    canvas_img = cv2.transpose(canvas_img)  # column-major order (like EMNIST)

    # Normalize and shape for model input
    canvas_img = canvas_img.astype(np.float32) / 255.0
    canvas_img = np.expand_dims(canvas_img, axis=(0, -1))  # shape: (1, 28, 28, 1)

    return canvas_img


def predict_character(img):
   
    pred = model.predict(img)
    label = np.argmax(pred)
    char = label_map.get(int(label),"?")  # fallback if missing
    return char

if __name__ == "__main__":
    tracker() 