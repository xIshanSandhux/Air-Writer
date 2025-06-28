import numpy as np
import cv2

class VirtualCanvas:
    def __init__(self, width=640, height=480):
        self.canvas = np.zeros((height, width), dtype=np.uint8)
        self.last_point = None

    def update(self, point):
        if point is None:
            self.last_point = None  # Reset when no hand detected
            return
        if self.last_point and point:
            cv2.line(self.canvas, self.last_point, point, 255, 4)
        self.last_point = point

    def clear(self):
        self.canvas.fill(0)
        self.last_point = None

    def get_image(self):
        return self.canvas.copy()
