#!/usr/bin/python

import os
import numpy as np
import cv2
import random
from glob import glob

# Image dimensions taken from squeezenet
WIDTH  = 227
HEIGHT = 227

'''
import shutil
import zipfile
from urllib2 import urlopen
from tqdm import tqdm

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_LARA_dataset(data_dir):
    """
    Download and extract LARA dataset if it doesn't exist
    :param data_dir: Directory to download the dataset to
    """
    LARA_filename = 'Lara_UrbanSeq1_JPG.zip'
    LARA_path = os.path.join(data_dir, 'Lara3D_UrbanSeq1_JPG')
    LARA_files = ['LARA_labels.txt', os.path.join(LARA_path, 'frame_000772.jpg')]

    missing_LARA_files = [LARA_file for LARA_file in LARA_files if not os.path.exists(LARA_file)]
    if missing_LARA_files:
        # Clean LARA dir
        if os.path.exists(LARA_path):
            shutil.rmtree(LARA_path)

        # Download LARA Dataset
        print('Downloading LARA dataset...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
        urlopen(
            'http://s150102174.onlinehome.fr/Lara_UrbanSeq1_JPG.zip',
            os.path.join(data_dir, LARA_filename))
            pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(data_dir, LARA_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(data_dir, LARA_filename))
'''



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

    # Initialize 2 numpy arrays for X_train and y_train
    indices = range(len(label_no))
    random.shuffle(indices)


    def get_batches_fn(batch_size):
	"""
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
	for batch_i in range(0, len(label_no), batch_size):
            images = []
            labels = []
            for index in indices[batch_i:batch_i+batch_size]:

                label = (label_class[index])
                if label == "'go'":
                    label = 2                 #'green'
                elif label == "'warning'":
                    label = 1                 #'yellow'
                elif label == "'stop'":
                    label = 0                 #'red'
                elif label == "'ambiguous'":
                    label = 4                 #'unknown'
                labels.append(label)

                for image_file in image_paths:
                    if label_no[index] == int(os.path.basename(image_file)[6:12]):
                        image = cv2.imread(image_file)
#                        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
#                        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
                        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_LINEAR)
                        images.append(image)
                        break

            # To visualise gen data
#            print(labels[0])
#            cv2.imshow("Image window", images[0])
#            cv2.waitKey(5000)

            yield np.array(images), np.array(labels)
    return get_batches_fn
