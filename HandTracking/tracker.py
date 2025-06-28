#using libraries mediapipe and cv2
# mediapipe is used to detect the hand and get the landmarks
# cv2 is used to display the frame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

class HandTracker:
    def __init__(self):
        # initialize the hands object
        # max_num_hands is the maximum number of hands to detect
        # min_detection_confidence is the minimum confidence score for a hand to be detected
        # min_tracking_confidence is the minimum confidence score for a hand to be tracked
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        # initialize the drawing utils object
        # drawing utils is used to draw the landmarks on the frame
        self.mp_draw = mp.solutions.drawing_utils

    def get_index_finger_tip(self, frame):
        # convert the frame to rgb
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the frame
        result = self.hands.process(rgb)

        # if there is a hand detected
        if result.multi_hand_landmarks:
            # get the first hand detected
            hand = result.multi_hand_landmarks[0]
            # index finger tip
            tip = hand.landmark[8] 
            # index finger mcp (knuckle)
            mcp = hand.landmark[5]

            if tip.y < mcp.y:
                h, w, _ = frame.shape
                return int(tip.x * w), int(tip.y * h)
            else:
                return None
    
    def middle_finger_up(self,frame):
        # convert the frame to rgb
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the frame
        result = self.hands.process(rgb)

        # if there is a hand detected
        if result.multi_hand_landmarks:
            # get the first hand detected
            hand = result.multi_hand_landmarks[0]
            # middle finger tip
            tip = hand.landmark[12] 
            # middle finger mcp (knuckle)
            mcp = hand.landmark[9]

            if tip.y < mcp.y:
                h, w, _ = frame.shape
                return int(tip.x * w), int(tip.y * h)
            else:
                return None
    