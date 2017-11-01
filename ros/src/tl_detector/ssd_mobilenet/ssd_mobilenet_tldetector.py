import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import click
import os
import glob
import collections

# Frozen inference graph files.
SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

RESULT_DIR = os.path.join(os.getcwd(),"test/")

# Colors (one for each class)
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

VIZ_DEBUG =0

COLOR_TO_CLASS = {"Red": 0,
                  "Yellow": 1,
                  "Green": 2,
                  "Unknown": 3}
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
    ''' Cropping image in a bounding boxes'''
    cropped_imgs=[]
    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = boxes[i, ...]
        print("Boxes: ", boxes[i, ...])
        w = (xmax- xmin)/2
        h = (ymax- ymin)/2
        cx = xmin + w
        cy = ymin + h

        print("cx %d,cy %d,w %d ,h %d  " %(cx,cy,w,h))
        cropped = image.crop( (xmin, ymin,  xmax, ymax))#(xmin, ymin, w, h))
        #croped_img = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        print(cropped.size)
        cropped_imgs.append(cropped)
        #cropped.save(RESULT_DIR+"cropped"+ str(i)+".jpg", "JPEG")
    return cropped_imgs

def match_histogram(cropped_imgs):
    ''' Classification using histogram color matching '''
    results=[]
    for cropped_img in cropped_imgs:
        w, h = cropped_img.size
        print("Mode: ", cropped_img.mode)
        # binary warped image has 3 channels
        np_cropped = np.array(cropped_img)
        print("cropped img shape: ",np_cropped.shape)
        RGB=[0.,0.,0.]
        # Show numpy histogram
        if len(np_cropped.shape) == 0:
            return

        for ch in range(3):
            flatten_ch0 = np_cropped[:,:,ch].flatten()
            color_freq=collections.Counter(flatten_ch0)

            total_val = np.sum(np.array(list(color_freq.values())))

            sorted_colors_tp = list(sorted(color_freq.items()))

            # calculate only the frequency of the 250-255 values
            for i in range(250,256):
                RGB[ch] = RGB[ch] + color_freq[i]

            # Normalization
            RGB[ch] = float(RGB[ch])/total_val

            if VIZ_DEBUG:
                if ch == 0:
                    strcolor="r"
                elif ch==1:
                    strcolor="g"
                else:
                    strcolor = "b"
                sorted_colors_tp = np.array(sorted_colors_tp)
                plt.bar(sorted_colors_tp[:, 0], sorted_colors_tp[:, 1]/total_val, color=strcolor, label="ch"+str(ch))
        if VIZ_DEBUG:
            plt.legend()
            plt.show()

        max_val = max(RGB)
        max_idx = [i for i, ch in enumerate(RGB) if ch == max_val]
        print("RGB: ",RGB)
        #print("Max idx : %s with val %f " % (max_idx, max_val))
        yellow_thres = 0.02


        if RGB[0] >yellow_thres and RGB[1] >yellow_thres and RGB[2] < 0.01:
            print("Yellow")
            results.append(COLOR_TO_CLASS["Yellow"])
        elif max_idx[0] == 0:
            print("Red")
            results.append(COLOR_TO_CLASS["Red"])
        elif max_idx[0] == 1:
            print("Green")
            results.append(COLOR_TO_CLASS["Green"])
        else:
            print("Unknown")
            results.append(COLOR_TO_CLASS["Unknown"])

        # Comparison to PIL histogram
        # ----------------------------------------
        # RGB_dict={"Red":0, "Yellow":0, "Green":0}
        # print(type(cropped_img.histogram()))
        # # https://stackoverflow.com/questions/22460577/understanding-histogram-in-pillow
        # for i, value in enumerate(cropped_img.histogram()):
        #     #print(i, value) # i: 0 -767
        #     if i > 0 and i <15:
        #         RGB_dict["Red"]= value
        #     elif i>500 and i < 515:
        #         RGB_dict["Green"]= value
        #     elif i >760:
        #         RGB_dict["Blue"]= value
        # #print(max(RGB_dict, key=RGB_dict.get))
        # if VIZ_DEBUG:
        #     plt.plot(cropped_img.histogram())
        #     plt.show()
    print("Final result: ", max(results))
    return max(results)

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

    def get_classification(self, image_path, DEBUG=False):
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

            confidence_cutoff = 0.4
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            width, height = image.size
            box_coords = to_image_coords(boxes, height, width)
            if DEBUG:
                print(scores)
                print("---")
                print(classes)
                print("boxes: ", boxes)
                print("Box coord: ",box_coords)

#            Cropped image
            if len(box_coords)> 0:
                image_np = np.squeeze(image_np)
                cropped_imgs = crop_image(image, box_coords)
                return match_histogram(cropped_imgs)

            if DEBUG:
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