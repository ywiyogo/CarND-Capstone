# Traffic Light classifier test
# Author: YWiyogo
# Usage: 1. copy the model files in to the model folder
#        2. run python no_ros_tl_classifier.py <input_image>
# Note: it works only for 1 input image file

import tensorflow as tf
import cv2
import os
import numpy as np
import click
import glob
#from scipy.misc import imresize

# Get the model directory
MODEL_DIR = os.getcwd()+ "/model"
LOG_DIR = os.getcwd()+ "/logs"

# Image dimensions taken from squeezenet
WIDTH  = 227
HEIGHT = 227

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

        print('--------------- Loading classifier --------------')
        print
        self.graph = tf.get_default_graph()

        with tf.Session() as self.sess:
        #http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        # Load the meta graph and restore weights
            self.saver = tf.train.import_meta_graph(MODEL_DIR + '/model.meta')

            print('---------------- Loading complete ---------------')


    def calc_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    def resize_image(self, image):
        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_LINEAR)
        return image

    def get_classification(self, image_path, debug=1):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        image = cv2.imread(image_path);
        self.saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_DIR))
        #print("graph: ",self.graph.get_operations())
        #print("self.saver: ", self.saver)

        #print("Image: ", image.shape)
        if image is None:
            print("Error: image is None!")
            return
        resized_img = self.resize_image(image)
        if debug:
            print("resized shape: ", resized_img.shape)



        mean_pixel = np.array([104.006, 116.669, 122.679], dtype=np.float32)
        #preproc_img = helper.preprocess(resized_img, mean_pixel)
        expanded_img = np.expand_dims(resized_img, axis=0)
        if debug:
            print('--------------- Getting classification --------------')


        # Placeholders
        relu_op = self.graph.get_tensor_by_name('Classifier/Relu_2:0')
        summary_writer = tf.summary.FileWriter(LOG_DIR, graph=self.sess.graph)

        # softmax = tf.nn.softmax(logits)

        # with tf.Session() as sess:
        #     output = sess.run(softmax, feed_dict={logits: logit_data})

        predictions = self.sess.run(relu_op, feed_dict=
                                        {"input_images:0": expanded_img,
                                         "keep_prob:0": 1.})
        if debug:
            print("Predictions: ", predictions)
        predictions = np.squeeze(predictions)   #squeeze array to 1 dim array

        if all(val==predictions[0] for val in predictions):
            print("File: %s; result: %d -> UNKNOWN result (0,0,0,0)" % (image_path, 3))

        softmax = self.calc_softmax(predictions)
        max_index = np.argmax(softmax)
        if debug:
            print("Softmax: ", softmax)
            print("argmax: ", max_index)
        else:
            print("File: %s; result: %d" % (image_path, max_index))

        return max_index
        print('--------------- Classification complete -------------')
        #return TrafficLight.UNKNOWN

@click.command()
@click.argument('img_path')
def main(img_path):
    tlc = TLClassifier()
    if os.path.isdir(img_path):
        imgfiles = img_path +"*.jpg"
        for img_file in sorted(glob.glob(imgfiles)):
            tlc.get_classification(img_file,0)
    else:
        tlc.get_classification(img_path)

if __name__ == '__main__':
    main()