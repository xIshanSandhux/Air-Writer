import numpy as np
import cv2

class VirtualCanvas:
    def __init__(self, width=400, height=400):
        self.canvas = np.zeros((height, width), dtype=np.uint8)
        self.last_point = None
        self.line_thickness = 8  # Thicker lines for better recognition

    def update(self, point):
        if point is None:
            self.last_point = None
            return
            
        if self.last_point and point:
            # Draw line between points
            cv2.line(self.canvas, self.last_point, point, (255,), self.line_thickness)
            
            # Add some smoothing by drawing circles at points
            cv2.circle(self.canvas, point, self.line_thickness//2, (255,), -1)
            cv2.circle(self.canvas, self.last_point, self.line_thickness//2, (255,), -1)
            
        self.last_point = point

    def clear(self):
        self.canvas.fill(0)
        self.last_point = None

    def get_image(self):
        return self.canvas.copy()
    
    def get_centered_image(self):
        """Get image with content centered and padded to square"""
        # Find bounding box of content
        coords = np.where(self.canvas > 0)
        if len(coords[0]) == 0:
            return np.zeros((400, 400), dtype=np.uint8)
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add padding
        padding = 20
        y_min = max(0, y_min - padding)
        y_max = min(self.canvas.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(self.canvas.shape[1], x_max + padding)
        
        # Extract content
        content = self.canvas[y_min:y_max, x_min:x_max]
        
        # Create square canvas
        size = max(content.shape[0], content.shape[1])
        square_canvas = np.zeros((size, size), dtype=np.uint8)
        
        # Center the content
        y_offset = (size - content.shape[0]) // 2
        x_offset = (size - content.shape[1]) // 2
        square_canvas[y_offset:y_offset+content.shape[0], x_offset:x_offset+content.shape[1]] = content
        
        return square_canvas
