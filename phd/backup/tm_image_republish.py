#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class TMImageRepublish(Node):
    def __init__(self):
        super().__init__("tm_image_republish")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/techman_image", self.cb, 10)
        self.pub = self.create_publisher(Image, "/techman_image_bgr8", 10)
        self.get_logger().info("Republishing /techman_image -> /techman_image_bgr8 (bgr8)")

    def cb(self, msg: Image):
        try:
            if msg.encoding == "8UC3":
                row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
                img = row[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
                out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out.header = msg.header
                self.pub.publish(out)
            else:
                # already standard-ish; pass through
                self.pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Convert/publish failed: {e}")

def main():
    rclpy.init()
    node = TMImageRepublish()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
