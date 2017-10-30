# Helper functions
# Base code is taken from https://github.com/fregu856/2D_detection

import cv2
import numpy as np
import tensorflow as tf
import os

def vis_anchors_vs_bboxes(img_path, img,  anchors, bboxes, gt_deltas=None):
    for box in bboxes:
        x_left = int(box[0] - box[2]/2)
        y_bottom = int(box[1] + box[3]/2)
        x_right = int(box[0] + box[2]/2)
        y_top = int(box[1] - box[3]/2)
        color=(0, 255, 255)
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom),
                    color, 2)

    for step, anchor in enumerate(anchors):
        if step < 300:
            x_left = int(anchor[0] - anchor[2]/2)
            y_bottom = int(anchor[1] + anchor[3]/2)
            x_right = int(anchor[0] + anchor[2]/2)
            y_top = int(anchor[1] - anchor[3]/2)
            color=(0, 0, 255)
            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom),
                        color, 2)
    print("GT deltas: ",gt_deltas)
    if not gt_deltas is None:
        for delta in gt_deltas:
            x_left = int(delta[0] - delta[2]/2)
            y_bottom = int(delta[1] + delta[3]/2)
            x_right = int(delta[0] + delta[2]/2)
            y_top = int(delta[1] - delta[3]/2)
            color=(250, 0, 0)
            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom),
                        color, 2)

    basename=os.path.basename(img_path)
    cv2.imshow(basename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_bbox_fcenter(shape_img_orig, shape_img_new, bbox):
    hscale = shape_img_new[0] / shape_img_orig[0]
    wscale = shape_img_new[1] / shape_img_orig[1]
    new_bbox = bbox
    new_bbox[0]= bbox[0]*wscale
    new_bbox[1]= bbox[1]*hscale
    new_bbox[2]= bbox[2]*wscale
    new_bbox[3]= bbox[3]*hscale
    return new_bbox

# Visualization with box format cx, cy, w, h, label
def vis_gt_bboxes(img_path, bboxes):
    class_label_to_string = {0: "Red", 1: "Yellow", 2: "Green", 3: "Off"}
    class_to_color = {"Red": (0, 0, 255),
                      "Yellow": (0, 255, 255),
                      "Green": (0,255,0),
                      "Off": (19, 139,69)}
    img = cv2.imread(img_path, -1)

    for box in bboxes:
        x_left = int(box[0] - box[2]/2)
        y_bottom = int(box[1] + box[3]/2)
        x_right = int(box[0] + box[2]/2)
        y_top = int(box[1] - box[3]/2)
        color=0
        if "Red" in box[4]:
            color=class_to_color["Red"]
        elif "Yellow" in box[4]:
            color=class_to_color["Yellow"]
        elif "Green" in box[4]:
            color=class_to_color["Green"]
        else:
            color = class_to_color["Off"]
        # draw the bbox:
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom),
                    color, 2)

    img_with_bboxes = img
    return img_with_bboxes

def vis_gt_bboxes_fminmax(img_path, bboxes):
    class_label_to_string = {0: "Red", 1: "Yellow", 2: "Green", 3: "Off"}
    class_to_color = {"Red": (0, 0, 255),
                      "Yellow": (0, 255, 255),
                      "Green": (0,255,0),
                      "Off": (19, 139,69)}
    img = cv2.imread(img_path, -1)

    for box in bboxes:
        print(box)
        xmin = int(box["x_min"])
        ymin = int(box["y_min"])
        xmax = int(box["x_max"])
        ymax = int(box["y_max"])
        color=0
        if "Red" in box["label"]:
            color=class_to_color["Red"]
        elif "Yellow" in box["label"]:
            color=class_to_color["Yellow"]
        elif "Green" in box["label"]:
            color=class_to_color["Green"]
        else:
            color = class_to_color["Off"]
        # draw the bbox:
        cv2.rectangle(img, (xmin, ymin), (xmax , ymax),
                    color, 2)

    img_with_bboxes = img
    return img_with_bboxes


# function for drawing all ground truth bboxes of an image on the image:
def visualize_gt_label(img_path, label_path):
    class_to_color = {"Red": (0,0, 255),
                      "Yellow": (128,0, 128),
                      "Green": (0,128,0),
                      "Off": (19, 139,69)}

    img = cv2.imread(img_path, -1)

    with open(label_path) as label_file:
        for line in label_file:
            splitted_line = line.split(" ")
            bbox_class = splitted_line[0].lower().strip()
            if bbox_class not in ["Red", "Yellow", "Green", "Off"]:
                break
            x_left = int(float(splitted_line[4]))
            y_bottom = int(float(splitted_line[5]))
            x_right = int(float(splitted_line[6]))
            y_top = int(float(splitted_line[7]))

            # draw the bbox:
            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom),
                        class_to_color[bbox_class], 2)

    img_with_bboxes = img
    return img_with_bboxes

# function for drawing a set of bboxes in an img:
def draw_bboxes(img, bboxes, class_labels, probs=None):
    class_label_to_string = {0: "Red", 1: "Yellow", 2: "Green", 3: "Off"}
    class_to_color = {"Red": (0, 0, 255),
                      "Yellow": (0, 255, 255),
                      "Green": (0,255,0),
                      "Off": (19, 139,69)}
    if probs is None:
        #ground truth
        probs=[]
        for i in range(len(bboxes)):
            probs.append(99)


    for bbox, class_label, prob in zip(bboxes, class_labels, probs):
        xmin, ymin, xmax, ymax = bbox_transform(bbox)

        h = ymax - ymin
        w = xmax - xmin

        class_string = class_label_to_string[class_label]

        # draw the bbox:
        cv2.rectangle(img, (int(xmin), int(ymax)), (int(xmax), int(ymin)),
                    class_to_color[class_string], 2)

        if probs is not None:
            # write the detection probability on the bbox:
            # # make the top line of the bbox thicker:
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymin-12)),
                        class_to_color[class_string], -1)
            # # write the probaility in the top line of the bbox:
            prob_string = "%.2f" % prob
            cv2.putText(img, prob_string, (int(xmin)+2, int(ymin)-2), 2, 0.4,
                        (255,255,255), 0)

    img_with_bboxes = img
    return img_with_bboxes

def safe_exp(w, thresh):
    # NOTE! this function is taken directly from
    # github.com/BichenWuUCB/squeezeDet

    slope = np.exp(thresh)

    lin_bool = w > thresh
    lin_region = tf.to_float(lin_bool)

    lin_out = slope*(w - thresh + 1.)
    exp_out = tf.exp(tf.where(lin_bool, tf.zeros_like(w), w))

    out = lin_region*lin_out + (1.-lin_region)*exp_out

    return out

# function for converting a bbox of [cx, cy, w, h] format to
# [xmin, ymin, xmax, ymax] format:
def bbox_transform(bbox):
    cx, cy, w, h = bbox

    xmin = cx - w/2
    ymin = cy - h/2
    xmax = cx + w/2
    ymax = cy + h/2

    out_box = [xmin, ymin, xmax, ymax]

    return out_box

# function for converting a bbox of [xmin, ymin, xmax, ymax] format to
# [cx, cy, w, h] format:
def bbox_transform_inv(bbox):
    xmin, ymin, xmax, ymax = bbox

    w = xmax - xmin + 1.0
    h = ymax - ymin + 1.0
    cx  = xmin + w/2
    cy  = ymin + h/2

    out_box = [cx, cy, w, h]

    return out_box

# function for performing non-maximum suppression:
def nms(boxes, probs, threshold):
    # NOTE! this function is taken directly from
    # github.com/BichenWuUCB/squeezeDet

    # get indices in descending order acc. to prob:
    order = probs.argsort()[::-1]

    keep = [True]*len(order)
    for i in range(len(order)-1):
        ovps = batch_IOU(boxes[order[i+1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j+i+1]] = False

    return keep

# function for computing the IOU between a bbox and a batch of bboxes:
def batch_IOU(boxes, box):

    intersect_xmax = np.minimum(boxes[:, 0] + 0.5*boxes[:, 2], box[0] + 0.5*box[2])
    intersect_xmin = np.maximum(boxes[:, 0] - 0.5*boxes[:, 2], box[0] - 0.5*box[2])
    intersect_ymax = np.minimum(boxes[:, 1] + 0.5*boxes[:, 3], box[1] + 0.5*box[3])
    intersect_ymin = np.maximum(boxes[:, 1] - 0.5*boxes[:, 3], box[1] - 0.5*box[3])

    intersect_w = np.maximum(0.0, intersect_xmax - intersect_xmin)
    intersect_h = np.maximum(0.0, intersect_ymax - intersect_ymin)
    intersection_area = intersect_w*intersect_h

    union_area = boxes[:, 2]*boxes[:, 3] + box[2]*box[3] - intersection_area

    IOUs = intersection_area/union_area

    return IOUs

# function for building a dense matrix from a sparse representation:
def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
    # NOTE! this function is a modified version of sparse_to_dense in
    # github.com/BichenWuUCB/squeezeDet

    # (indices: list of indices. if indices[i] = [k, l], then array[k,l] should
    # be set to values[i])

    # (output_shape: shape of the dense matrix)

    # (values: list of values. if indices[i] = [k, l], then array[k,l] should be
    # set to values[i])

    # (default_value: values to set for indices not specified in indices)
    assert len(sp_indices) == len(values), 'Length of sp_indices is not equal to length of values'

    array = np.ones(output_shape) * default_value
    for idx, value in zip(sp_indices, values):
        array[tuple(idx)] = value
    return array
