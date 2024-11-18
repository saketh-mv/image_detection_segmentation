from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the YOLO model for segmentation
model_path = os.path.join(os.path.dirname(__file__),"yolov11_seg_final.pt")
model = YOLO(model_path)

# Function to perform segmentation on a single image
def segment_objects(image: np.ndarray) -> np.ndarray:
    """
    Perform segmentation on an image using YOLOv11 and return the annotated image with masks.
    
    Args:
        image (np.ndarray): The input image to be processed.

    Returns:
        np.ndarray: The image with detected segmentation masks and annotated bounding boxes.
    """
    # Perform segmentation on the image
    results = model.predict(source=image)
    
    annotated_image = results[0].plot()  # This provides the annotated image with masks

    return annotated_image

# Example usage:
if __name__ == "__main__":
    # Load an example image using OpenCV
    image_path = "/app/yolo_seg/small_test/13.jpg"
    image = cv2.imread(image_path)

    # Perform segmentation on the image
    annotated_image = segment_objects(image)

    # Display the annotated image
    cv2.imshow("Segmented Objects", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
