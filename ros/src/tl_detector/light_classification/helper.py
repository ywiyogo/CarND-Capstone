#!/usr/bin/python

from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imresize
from glob import glob
import numpy as np
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

batch_size = 64


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
    labels_path = os.path.join(data_path, 'all_label.txt')
    # Read in label information
    data = np.loadtxt(labels_path, dtype=str, delimiter=' ')
    np.random.shuffle(data)

    l = len(data)
    train_data = data[0:int(l*RATIO)]
    test_data  = data[int(l*RATIO)+1:l]

    def get_batches_fn():
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        for batch_i in range(0, len(train_data), batch_size):
            X_train = []
            y_train = []

            for row in train_data[batch_i:batch_i+batch_size]:

                label = int(row[2])
                image = (get_image(row[3]))

                y_train.append(label)
                X_train.append(image)

            # Augment images
            X_train, y_train = augment_images(X_train, y_train)
            # Yield
            yield np.array(X_train), np.array(y_train)


    # Get X_test and y_test
    print('Generating test set... {}% train, {}% testing'.format(RATIO*100, (1-RATIO)*100))
    X_test = []
    y_test = []
    for r in test_data:
        X_test.append(get_image(r[3]))
        y_test.append(int(r[2]))

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
        np.random.shuffle(bosch_data)

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
            images, labels = augment_images(images, labels)
            # Yield
            yield np.array(images), np.array(labels)


    # Get X_test and y_test
    #print('Generating test set... {}% train, {}% testing'.format(RATIO*100, (1-RATIO)*100))
    X_test = []
    y_test = []

    return get_batches_fn, np.array(X_test), np.array(y_test)

def augment_images(X, y):

    def flip_lr(X_):
        X_ = np.fliplr(X_)
        return X_

    def brighten(X_):
        num = float(np.random.randint(40) - 20)
        if (num >= 0): num += 10
        else: num -= 10
        X_ = cv2.add(X_, np.array([num]))
        return X_

    def translation(X_, rows, cols):
        tr_x = 20*np.random.uniform() - 10
        tr_y = 20*np.random.uniform() - 10
        Trans_M = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
        X_ = cv2.warpAffine(X_, Trans_M, (cols, rows))
        return X_

    def shear(X_, rows, cols):
        shear_range = 2
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range*np.random.uniform() - shear_range/2
        pt2 = 20 + shear_range*np.random.uniform() - shear_range/2
        pts2 = np.float32([[pt1, 5],[pt2, pt1],[5, pt2]])
        shear_M = cv2.getAffineTransform(pts1, pts2)
        X_ = cv2.warpAffine(X_, shear_M, (cols, rows))
        return X_

    for i in range(len(y)):
        X_ = np.copy(X[i])
        y_ = y[i]
        rows, cols, _ = X_.shape
        if (np.random.randint(2) == 1): X_ = flip_lr(X_)
        if (np.random.randint(2) == 1): X_ = brighten(X_)
        if (np.random.randint(2) == 1): X_ = translation(X_, rows, cols)
        if (np.random.randint(2) == 1): X_ = shear(X_, rows, cols)
#        print('Label:', y_)
#        cv2.imshow('original', X[i])
#        cv2.imshow('augmented', X_)
#        cv2.waitKey()
        X[i] = X_
        y[i] = y_
    return X, y