from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the YOLO model
model_path = os.path.join(os.path.dirname(__file__), "yolov11_final.pt")
model = YOLO(model_path)

# Function to detect objects on a single image
def detect_objects(image: np.ndarray) -> np.ndarray:
    """
    Detect objects in an image using YOLOv11 and return the annotated image.
    
    Args:
        image (np.ndarray): The input image to be processed.

    Returns:
        np.ndarray: The image with detected objects annotated.
    """
    # Perform object detection on the image
    results = model.predict(source=image)
    
    # Render the results on the image
    annotated_image = results[0].plot()  # This provides the annotated image
    
    return annotated_image

# Example usage:
if __name__ == "__main__":
    # Load an example image using OpenCV
    image_path = "/app/yolo/train/12.jpg"
    image = cv2.imread(image_path)

    # Detect objects in the image
    annotated_image = detect_objects(image)

    # Display the annotated image
    cv2.imshow("Detected Objects", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

