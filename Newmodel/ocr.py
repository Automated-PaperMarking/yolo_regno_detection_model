# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO

# try:
#     # Load YOLO model
#     yolo_model = YOLO(r".\my_model\train\weights\best.pt")
    
#     # Load image
#     image = cv2.imread(r".\test2.jpg")
    
#     if image is None:
#         raise ValueError("Could not load image. Check the file path.")
    
#     # Display original image
#     plt.figure(figsize=(12, 8))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#     plt.axis("off")
#     plt.show()

#     # Run YOLO detection
#     results = yolo_model(image)
    
#     # Get detection results
#     detections = results[0]
#     boxes = detections.boxes
    
#     if boxes is not None and len(boxes) > 0:
#         print(f"Found {len(boxes)} digit detections")
        
#         # Process each detected digit
#         detected_digits = []
        
#         for i, box in enumerate(boxes):
#             # Get bounding box coordinates
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#             confidence = box.conf[0].cpu().numpy()
#             class_id = int(box.cls[0].cpu().numpy())
            
#             # Get class name (digit) from the model
#             class_name = yolo_model.names[class_id]
#             print(f"Detection {i+1}: Digit '{class_name}' at Box({x1},{y1},{x2},{y2}), Confidence: {confidence:.3f}")
#             detected_digits.append({
#                 'digit': class_name,
#                 'confidence': confidence,
#                 'bbox': (x1, y1, x2, y2),
#                 'x_center': (x1 + x2) // 2
#             })
        
#         # Sort detections by x-coordinate (left to right)
#         detected_digits.sort(key=lambda x: x['x_center'])
        
#         # Create visualization
#         result_image = image.copy()
        
#         # Draw bounding boxes and predictions
#         for i, detection in enumerate(detected_digits):
#             x1, y1, x2, y2 = detection['bbox']
#             digit = detection['digit']
#             conf = detection['confidence']
            
#             # Draw bounding box (green)
#             cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#             # Add text with prediction
#             text = f"{digit} ({conf:.2f})"
#             cv2.putText(result_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             print(f"Position {i+1}: Digit '{digit}', Confidence: {conf:.3f}")
        
#         # Display result
#         plt.figure(figsize=(12, 8))
#         plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
#         if len(detected_digits) < 10:
#             print(f"Warning: Only {len(detected_digits)} digits detected. Some positions may be missing.")
#         plt.show()
        
#         # Print final sequence
#         digit_sequence = [d['digit'] for d in detected_digits]
#         print(f"\nDetected digit sequence (left to right): {' '.join(digit_sequence)}")
#         print(f"Complete number: {''.join(digit_sequence)}")
        
#     else:
#         print("No digits detected in the image")
        
#         # Display original image anyway
#         plt.figure(figsize=(12, 8))
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title("No Digits Detected")
#         plt.axis("off")
#         plt.show()

# except Exception as e:
#     print(f"Error during processing: {e}")
#     import traceback
#     traceback.print_exc()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def detect_digits(image_path, model_path, show_plots=True):
    """
    Detect digits in an image using YOLO model with validations.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the YOLO model
        show_plots (bool): Whether to display visualization plots
    
    Returns:
        dict: Detection results with validation messages
    """
    try:
        # Load YOLO model
        yolo_model = YOLO(model_path)
        
        # Load image
        image = cv2.imread(image_path)
    
        if image is None:
            return {"success": False, "error": "Could not load image. Check the file path."}
        
        # Display original image
        if show_plots:
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
        
        result_data = {
            "success": True,
            "total_detections": 0,
            "detected_digits": [],
            "digit_sequence": "",
            "complete_number": "",
            "validation_messages": [],
            "confidence_assessment": ""
        }
        
        if boxes is not None and len(boxes) > 0:
            print(f"Found {len(boxes)} digit detections")
            result_data["total_detections"] = len(boxes)
            
            # Process each detected digit
            detected_digits = []
        
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name (digit) from the model
                class_name = yolo_model.names[class_id]
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                
                print(f"Detection {i+1}: Digit '{class_name}' at Box({x1},{y1},{x2},{y2}), Center({x_center},{y_center}), Confidence: {confidence:.3f}")
                
                detected_digits.append({
                    'digit': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'x_center': x_center,
                    'y_center': y_center
                })
        
            # Validation 1: Check center point positions (y-coordinate should not be less than 200)
            low_position_digits = [d for d in detected_digits if d['y_center'] < 200]
            if low_position_digits:
                message = f"Warning: {len(low_position_digits)} digit(s) detected with center point below 200 pixels (too high in image). This may indicate incorrect detection."
                result_data["validation_messages"].append(message)
                print(f"\n{message}")
                for digit in low_position_digits:
                    print(f"  - Digit '{digit['digit']}' at center ({digit['x_center']}, {digit['y_center']})")
            
            # Sort detections by x-coordinate (left to right)
            detected_digits.sort(key=lambda x: x['x_center'])
            result_data["detected_digits"] = detected_digits
            
            # Validation 2: Check if total digits equals 8
            if len(detected_digits) != 8:
                if len(detected_digits) < 8:
                    message = f"Warning: Only {len(detected_digits)} digits detected. Expected 8 digits. Some digits may be missing or not detected properly."
                else:
                    message = f"Warning: {len(detected_digits)} digits detected. Expected 8 digits. There may be false positive detections."
                result_data["validation_messages"].append(message)
                print(f"\n{message}")
            else:
                message = "✓ Correct number of digits detected (8 digits)."
                result_data["validation_messages"].append(message)
                print(f"\n{message}")
            
            # Validation 3: Confidence assessment
            confidences = [d['confidence'] for d in detected_digits]
            avg_confidence = np.mean(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            if avg_confidence >= 0.8:
                confidence_level = "HIGH"
                confidence_comment = f"High confidence detections (avg: {avg_confidence:.3f}). Results are likely accurate."
            elif avg_confidence >= 0.6:
                confidence_level = "MEDIUM"
                confidence_comment = f"Medium confidence detections (avg: {avg_confidence:.3f}). Results are moderately reliable."
            else:
                confidence_level = "LOW"
                confidence_comment = f"Low confidence detections (avg: {avg_confidence:.3f}). Results may be unreliable, consider better image quality."
            
            result_data["confidence_assessment"] = confidence_comment
            print(f"\nConfidence Assessment: {confidence_level}")
            print(f"Average confidence: {avg_confidence:.3f} (range: {min_confidence:.3f} - {max_confidence:.3f})")
            print(confidence_comment)
            
            # Create visualization
            result_image = image.copy()
        
            # Draw bounding boxes and predictions
            for i, detection in enumerate(detected_digits):
                x1, y1, x2, y2 = detection['bbox']
                digit = detection['digit']
                conf = detection['confidence']
                
                # Choose color based on confidence and position validation
                if detection['y_center'] < 200:
                    color = (0, 0, 255)  # Red for low position
                elif conf >= 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif conf >= 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (255, 0, 0)  # Blue for low confidence
                
                # Draw bounding box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # Add text with prediction
                text = f"{digit} ({conf:.2f})"
                cv2.putText(result_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                print(f"Position {i+1}: Digit '{digit}', Confidence: {conf:.3f}")
        
            # Display result
            if show_plots:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                plt.title("Digit Detection Results")
                plt.axis("off")
                plt.show()
            
            # Generate final sequence
            digit_sequence = [d['digit'] for d in detected_digits]
            complete_number = ''.join(digit_sequence)
            
            result_data["digit_sequence"] = ' '.join(digit_sequence)
            result_data["complete_number"] = complete_number
            
            print(f"\nDetected digit sequence (left to right): {' '.join(digit_sequence)}")
            print(f"Complete number: {complete_number}")
        
        else:
            result_data["validation_messages"].append("No digits detected in the image")
            print("No digits detected in the image")
            
            # Display original image anyway
            if show_plots:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title("No Digits Detected")
                plt.axis("off")
                plt.show()
        
        return result_data
        
    except Exception as e:
        return {"success": False, "error": f"Error during processing: {e}"}

# Example usage
if __name__ == "__main__":
    try:
        model_path = r".\my_model\train\weights\best.pt"
        image_path = r".\test2.jpg"
        
        result = detect_digits(image_path, model_path, show_plots=True)
        
        if result["success"]:
            print("\n" + "="*50)
            print("DETECTION SUMMARY")
            print("="*50)
            print(f"Total detections: {result['total_detections']}")
            print(f"Complete number: {result['complete_number']}")
            print(f"Confidence assessment: {result['confidence_assessment']}")
            
            if result['validation_messages']:
                print("\nValidation Messages:")
                for msg in result['validation_messages']:
                    print(f"- {msg}")
        else:
            print(f"Detection failed: {result['error']}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
