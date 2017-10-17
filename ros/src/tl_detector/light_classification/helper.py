#!/usr/bin/python

from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imresize
from glob import glob
import numpy as np
import random
import math
import yaml
import os
import re
import cv2

# Image dimensions taken from squeezenet
WIDTH  = 227
HEIGHT = 227
# Ratio of training to test data [0:1]
RATIO  = 0.8

batch_size = 128


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
    image = imread(image_file, mode='RGB')
    image = resize_image(image)
    # Change from BGR to RGB
    image = image[...,::-1]
    return image

def resize_image(image):
    image = imresize(image, (WIDTH, HEIGHT))
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
    #image_paths = glob(os.path.join(data_path, 'Lara3D_UrbanSeq1_JPG/*.jpg'))
    #labels_path = os.path.join(data_path, 'LARA_labels.txt')
    image_paths = glob(os.path.join(data_path, 'all_dataset')+"/*")
    labels_path = os.path.join(data_path, 'all_label.txt')
    # Read in label information
    label_no    = np.loadtxt(labels_path, dtype=int, delimiter=' ', skiprows=1, usecols=(1,))
    label_class = np.loadtxt(labels_path, dtype=str, delimiter=' ', skiprows=1, usecols=(2,))

    # Split labels into train and test
    l = len(label_no)
    indices = np.arange(l)
    random.shuffle(indices)
    train_indices = indices[0:int(l*RATIO)]
    test_indices  = indices[int(l*RATIO)+1:l]

    def get_batches_fn():
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        # Shuffle training data for each epoch
        random.shuffle(train_indices)

        for batch_i in range(0, len(train_indices), batch_size):
            images = []
            labels = []

            for index in train_indices[batch_i:batch_i+batch_size]:
                label = (label_class[index])
                labels.append(label)
                for image_file in image_paths:
                    pattern=".*_0*"+str(label_no[index])+"\."
                    if re.match(pattern, image_file):
                        images.append(get_image(image_file))
                        break

            # Augment images
            images, labels = flip_lr(images, labels)
            # Yield
            yield np.array(images), np.array(labels)


    # Get X_test and y_test
    print('Generating test set... {}% train, {}% testing'.format(RATIO*100, (1-RATIO)*100))
    X_test = []
    y_test = []

    for index in test_indices:
        label = (label_class[index])
        y_test.append(label)

        for image_file in image_paths:
            pattern=".*_0*"+str(label_no[index])+"\."
            if re.match(pattern, image_file):
                X_test.append(get_image(image_file))
                break

    return get_batches_fn, np.array(X_test), np.array(y_test)

def gen_batch_function_Bosch(data_path):
    """
    Generate function to create batches of training data
    :param data_path: Path to folder that train.yaml

    ################
    # Bosch Dataset #
    ################

    """
    # Paths to relative files
    input_yaml = os.path.join(data_path, 'BOSCH/train.yaml')

    bosch_data = yaml.load(open(input_yaml, 'rb').read())

    def get_batches_fn():
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        # Shuffle training data for each epoch
        random.shuffle(bosch_data)

        for batch_i in range(0, len(bosch_data), batch_size):

            images = []
            labels = []

            for image_dict in bosch_data[batch_i:batch_i+batch_size]:
                
                label = 3

                image_path = os.path.abspath(os.path.join(os.path.dirname(input_yaml), image_dict['path']))
                image = get_image(image_path)
                if image is None:
                    raise IOError('Could not open image path', image_dict['path'])            

                for box in image_dict['boxes']:
                    if box['label'] == 'Red':
                        label = 0
                        break
                    elif box['label'] == 'Yellow':
                        label = 1
                        break
                    elif box['label'] == 'Green':
                        label = 2
                        break

                images.append(image)
                labels.append(label)

#                print('>>>>>', label)
#                cv2.imshow('labeled_image', image)
#                cv2.waitKey(3000)

            # Augment images
            images, labels = flip_lr(images, labels)
            # Yield
            yield np.array(images), np.array(labels)


    # Get X_test and y_test
    #print('Generating test set... {}% train, {}% testing'.format(RATIO*100, (1-RATIO)*100))
    X_test = []
    y_test = []

    return get_batches_fn, np.array(X_test), np.array(y_test)

def flip_lr(X, y):
    for i in range(len(y)):
        if not (y == 0):
            X.append(np.fliplr(X[i]))
            y.append(y[i])

    return X, y