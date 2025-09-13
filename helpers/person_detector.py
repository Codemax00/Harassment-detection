# Simple person detector using YOLO
# This file helps find people in images

import cv2
from ultralytics import YOLO

def load_person_detector():
    """Load the YOLO model to detect people"""
    print("Loading person detector...")
    
    # Fix for PyTorch 2.6+ compatibility
    import torch
    
    # Set weights_only to False for compatibility
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
    
    try:
        model = YOLO('yolov8n.pt')  # Small, fast model
        print("Person detector loaded!")
        return model
    finally:
        # Restore original torch.load
        torch.load = original_load

def find_people_in_frame(model, frame):
    """Find all people in a video frame"""
    # Run YOLO detection
    results = model(frame)
    
    people_found = []
    
    # Look through all detected objects
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Class 0 is 'person' in YOLO
                if int(box.cls[0]) == 0:  
                    # Get coordinates of person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Only keep detections we're confident about
                    if confidence > 0.5:
                        people_found.append({
                            'x1': int(x1),
                            'y1': int(y1), 
                            'x2': int(x2),
                            'y2': int(y2),
                            'confidence': confidence
                        })
    
    return people_found

def draw_person_boxes(frame, people_list):
    """Draw boxes around detected people"""
    for person in people_list:
        x1, y1, x2, y2 = person['x1'], person['y1'], person['x2'], person['y2']
        confidence = person['confidence']
        
        # Draw green box around person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence text
        text = f"Person: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame
