import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    # Load YOLO model
    yolo_model = YOLO(r".\my_model\train\weights\best.pt")
    
    # Load image
    image = cv2.imread(r".\testdata\test.jpeg")
    
    if image is None:
        raise ValueError("Could not load image. Check the file path.")
    
    # Display original image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # Run YOLO detection
    results = yolo_model(image)
    
    # Get detection results
    detections = results[0]
    boxes = detections.boxes
    
    if boxes is not None and len(boxes) > 0:
        print(f"Found {len(boxes)} digit detections")
        
        # Process each detected digit
        detected_digits = []
        
        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name (digit) from the model
            class_name = yolo_model.names[class_id]
            print(f"Detection {i+1}: Digit '{class_name}' at Box({x1},{y1},{x2},{y2}), Confidence: {confidence:.3f}")
            detected_digits.append({
                'digit': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'x_center': (x1 + x2) // 2
            })
        
        # Sort detections by x-coordinate (left to right)
        detected_digits.sort(key=lambda x: x['x_center'])
        
        # Create visualization
        result_image = image.copy()
        
        # Draw bounding boxes and predictions
        for i, detection in enumerate(detected_digits):
            x1, y1, x2, y2 = detection['bbox']
            digit = detection['digit']
            conf = detection['confidence']
            
            # Draw bounding box (green)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text with prediction
            text = f"{digit} ({conf:.2f})"
            cv2.putText(result_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"Position {i+1}: Digit '{digit}', Confidence: {conf:.3f}")
        
        # Display result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        if len(detected_digits) < 10:
            print(f"Warning: Only {len(detected_digits)} digits detected. Some positions may be missing.")
        plt.show()
        
        # Print final sequence
        digit_sequence = [d['digit'] for d in detected_digits]
        print(f"\nDetected digit sequence (left to right): {' '.join(digit_sequence)}")
        print(f"Complete number: {''.join(digit_sequence)}")
        
    else:
        print("No digits detected in the image")
        
        # Display original image anyway
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("No Digits Detected")
        plt.axis("off")
        plt.show()

except Exception as e:
    print(f"Error during processing: {e}")
    import traceback
    traceback.print_exc()
