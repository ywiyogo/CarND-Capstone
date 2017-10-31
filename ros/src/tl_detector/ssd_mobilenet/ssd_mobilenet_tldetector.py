import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm
import click
import os
import glob
from scipy.misc import imread
from scipy.misc import imshow, imsave
from scipy.misc import imresize

# Frozen inference graph files.
SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

RESULT_DIR = os.path.join(os.getcwd(),"test/")

# Colors (one for each class)
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

#
# Utility funcs
#

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            if classes[i] == 10:
                idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def crop_image(image, boxes):
    for i in range(len(boxes)):
        ymax, xmin, ymin, xmax = boxes[i, ...]
        w = (xmax- xmin)/2
        h = (ymax- ymin)/2
        cx = xmin + w
        cy = ymin + h

        cropped = image.crop( (ymax, xmin,  xmax, ymin))#(xmin, ymin, w, h))
        #croped_img = image[int(ymin):int(ymax), int(xmin):int(xmax)]

        cropped.save(RESULT_DIR+"cropped"+ str(i)+".jpg", "JPEG")


class TLDetector(object):
    def __init__(self):
        self.detection_graph = load_graph(SSD_GRAPH_FILE)
        # detection_graph = load_graph(RFCN_GRAPH_FILE)
        # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def get_classification(self, image_path, DEBUG=1):
        image = Image.open(image_path)
        image_np = np.expand_dims(image, 0)
        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})
            summary_writer = tf.summary.FileWriter("./logs", sess.graph)

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            print(scores)
            print("---")
            print(classes)
            confidence_cutoff = 0.4
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            width, height = image.size
            box_coords = to_image_coords(boxes, height, width)
            print(box_coords)

            # Cropped image
            image_np = np.squeeze(image_np)
            print("after squeeze: ", image_np.shape)
            crop_image(image, box_coords)


            # Each class with be represented by a differently colored box
            draw_boxes(image, box_coords, classes)
            basename = os.path.basename(image_path)

            image.save(RESULT_DIR+basename, "JPEG")


            #plt.figure(figsize=(12, 8))
            #plt.imshow(image)


@click.command()
@click.argument('img_path')
def main(img_path):
    tlc = TLDetector()
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if os.path.isdir(img_path):
        imgfiles = img_path +"/*.jpg"

        for img_file in sorted(glob.glob(imgfiles)):

            tlc.get_classification(img_file, 0)
    else:
        tlc.get_classification(img_path)

if __name__ == '__main__':
    main()