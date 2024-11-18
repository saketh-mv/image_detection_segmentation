import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Import object detection and segmentation modules
from object_detection_segmentation.detection.detector import detect_objects
from object_detection_segmentation.segment.segmentor import segment_objects

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.get_logger().info('Received an image')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Perform object detection (YOLO)
        detected_objects = detect_objects(cv_image)

        # Perform segmentation
        segmented_image = segment_objects(cv_image)

        # Show results for visualization
        cv2.imshow('Detected Objects', detected_objects)
        cv2.imshow('Segmented Image', segmented_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    processor = ImageProcessor()
    rclpy.spin(processor)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

