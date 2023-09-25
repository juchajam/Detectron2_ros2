#! /usr/bin/env python3

import sys
import threading
import time

import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from cv_bridge import CvBridge, CvBridgeError
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2_msgs.msg import DetectResult
from sensor_msgs.msg import Image, RegionOfInterest


class Detectron2node(Node):
    def __init__(self, node_name='detectron2_ros'):
        super().__init__(node_name=node_name)
        self.get_logger().info("Initializing")
        setup_logger()

        self.declare_parameter('input_topic', rclpy.Parameter.Type.STRING)
        self.declare_parameter('detection_threshold', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('config', rclpy.Parameter.Type.STRING)
        self.declare_parameter('model', rclpy.Parameter.Type.STRING)
        self.declare_parameter('visualization', rclpy.Parameter.Type.BOOL)

        self._bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()
        self._image_counter = 0

        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.load_param('config'))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.load_param('detection_threshold') # set threshold for this model
        self.cfg.MODEL.WEIGHTS = self.load_param('model')
        self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)
        self._class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

        self._visualization = self.load_param('visualization',True)
        self._result_pub = self.create_publisher(DetectResult, 'result', 1)
        self._vis_pub = self.create_publisher(Image, 'visualization', 1)
        input_topic = self.load_param('input_topic')
        if input_topic:
            self._sub = self.create_subscription(Image, input_topic, self.callback_image, 1)
        self.start_time = time.time()
        self.timer = self.create_timer(0.01, self.run)
        self.get_logger().info("Initialized")

    def run(self):
        if self._msg_lock.acquire(False):
            img_msg = self._last_msg
            self._last_msg = None
            self._msg_lock.release()
        else:
            return

        if img_msg is not None:
            self._image_counter = self._image_counter + 1
            if (self._image_counter % 11) == 10:
                self.get_logger().info(f"Images detected per second={float(self._image_counter) / (time.time() - self.start_time):.2f}")

            np_image = self.convert_to_cv_image(img_msg)

            outputs = self.predictor(np_image)
            result = outputs["instances"].to("cpu")
            result_msg = self.getResult(result)

            if result_msg is not None:
                self._result_pub.publish(result_msg)

            # Visualize results
            if self._visualization:
                v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img = v.get_image()[:, :, ::-1]

                image_msg = self._bridge.cv2_to_imgmsg(img, encoding='rgb8')
                self._vis_pub.publish(image_msg)
                

    def getResult(self, predictions):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
        else:
            return

        result_msg = DetectResult()
        result_msg.header = self._header
        if predictions.has("pred_classes"):
            result_msg.class_ids = predictions.pred_classes.tolist()
            result_msg.class_names = np.array(self._class_names)[predictions.pred_classes.numpy()].tolist()
            
        if predictions.has("scores"):
            result_msg.scores = predictions.scores.tolist()

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self._bridge.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1).item()
            box.y_offset = np.uint32(y1).item()
            box.height = np.uint32(y2 - y1).item()
            box.width = np.uint32(x2 - x1).item()
            result_msg.boxes.append(box)

        return result_msg

    def convert_to_cv_image(self, image_msg):
        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

        return cv_img

    def callback_image(self, msg):
        self.get_logger().debug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._header = msg.header
            self._msg_lock.release()

    def load_param(self, param, default=None):
        new_param = self.get_parameter(param)
        self.get_logger().info(f"{param}: {new_param.value}")
        return new_param.value

def main():
    rclpy.init()
    node = Detectron2node()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
