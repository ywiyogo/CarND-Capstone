#!/usr/bin/python

import os
import numpy as np
import cv2
import math
import random
from glob import glob


# Image dimensions taken from squeezenet
WIDTH  = 227
HEIGHT = 227
# Ratio of training to test data [0:1]
RATIO  = 0.8

batch_size = 2


def get_class(label):
    '''
    :param new_label: Is the class index
    '''
    if label == "'stop'":
        new_label = 0                 #'red'
    elif label == "'warning'":
        new_label = 1                 #'yellow'
    elif label == "'go'":
        new_label = 2                 #'green'
    elif label == "'ambiguous'":
        new_label = 3                 #'unknown'
    return new_label

def get_image(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_LINEAR)
#    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
#    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
    return image


def gen_batch_function_LARA(data_path):
    """
    Generate function to create batches of training data
    :param data_path: Path to folder that contains all the datasets and labels

    ################
    # LARA Dataset #
    ################

    """
    # Paths to relative files
    image_paths = glob(os.path.join(data_path, 'Lara3D_UrbanSeq1_JPG/*.jpg'))
    labels_path = os.path.join(data_path, 'LARA_labels.txt')

    # Read in label information
    label_no    = np.loadtxt(labels_path, dtype=int, delimiter=' ', skiprows=1, usecols=(2,))
    label_class = np.loadtxt(labels_path, dtype=str, delimiter=' ', skiprows=1, usecols=(10,))

    # Split labels into train and test
    l = len(label_no)
    indices = np.arange(l)
    random.shuffle(indices)
    train_indices = indices[0:int(l*RATIO)]
    test_indices  = indices[int(l*RATIO)+1:l]

    def get_batches_fn(batch_size):
	"""
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        length = int(math.floor(len(train_indices)/batch_size))
        for batch_i in range(0, length*batch_size, batch_size):
            images = []
            labels = []

            for index in train_indices[batch_i:batch_i+batch_size]:
                label = (label_class[index])
                labels.append(get_class(label))

                for image_file in image_paths:
                    if label_no[index] == int(os.path.basename(image_file)[6:12]):
                        images.append(get_image(image_file))
                        break

            # To visualise gen data
#            print(labels[0])
#            cv2.imshow("Image window", images[0])
#            cv2.waitKey(5000)

            yield np.array(images), np.array(labels)


    # Get X_test and y_test
    print('Generating test set... {}'.format(RATIO))
    X_test = []
    y_test = []

    for index in test_indices:
        label = (label_class[index])
        y_test.append(get_class(label))

        for image_file in image_paths:
            if label_no[index] == int(os.path.basename(image_file)[6:12]):
                X_test.append(get_image(image_file))
                break
    return get_batches_fn, np.array(X_test), np.array(y_test)
