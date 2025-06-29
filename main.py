import cv2
from HandTracking.tracker import HandTracker
from canvas.drawing import VirtualCanvas
from tensorflow.keras.models import load_model
import numpy as np
from data.mapping import get_emnist_mapping
import time


label_map = get_emnist_mapping()
model = load_model("saved_models/emnist_cnn_model.h5")
def AirWriter():
    
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    canvas = VirtualCanvas(640,480)
    text_output = ""
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            processed_frame = frame.copy()
            if ret:
                frame = cv2.flip(frame, 1)
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Get index finger tip
            index_tip = tracker.get_index_finger_tip(frame)
            

            # Get middle finger tip
            middle_tip = tracker.middle_finger_up(frame)
            
            canvas.update(index_tip)
            
            if index_tip:
                x, y = index_tip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"Tip: ({x}, {y})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                canvas.update((x, y))
                if middle_tip:
                    canvas_img = canvas.get_centered_image()  # Use centered image
                    preprocessed = preprocess_canvas(canvas_img)
                    letter = predict_character(preprocessed)
                    text_output += letter
                    canvas.clear()
                    time.sleep(0.5)
                    print(f" Predicted letter: {letter}")
                    print(f" Text output: {text_output}")

                    
                    # time.sleep(1)
            else:
                cv2.putText(frame, "No index finger tip detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # canvas_bgr = cv2.cvtColor(canvas.get_image(), cv2.COLOR_GRAY2BGR)
            # combined = cv2.addWeighted(frame, 1, canvas_bgr, 0.5, 0)
            
            cv2.putText(frame, f"Output: {text_output}", (10, 450), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
            cv2.imshow('Hand Tracker', frame)
            cv2.imshow('Canvas', canvas.get_image())
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def preprocess_canvas(canvas_img):
    """
    Preprocesses the drawn canvas to match EMNIST Letters input format.
    """
    
    if len(canvas_img.shape) == 3:
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)

    
    canvas_img = cv2.resize(canvas_img, (28, 28), interpolation=cv2.INTER_AREA)

    
    canvas_img = canvas_img.astype(np.float32) / 255.0
   
    # Training data: reshape((-1, 28, 28), order="F").transpose(0, 2, 1)
    # For single image: reshape((28, 28), order="F").transpose(1, 0)
    canvas_img = canvas_img.reshape((28, 28), order="F").transpose(1, 0)

   
    canvas_img = np.expand_dims(canvas_img, axis=(0, -1))  # shape: (1, 28, 28, 1)

    return canvas_img


def predict_character(img):
   
    pred = model.predict(img)
    label = np.argmax(pred)
    char = label_map.get(int(label),"?")  # fallback if missing
    return char

if __name__ == "__main__":
    AirWriter() 