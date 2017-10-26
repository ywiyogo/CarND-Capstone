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
from model import SqueezeDet_model
import pickle
from utilities import draw_bboxes

# Get the model directory
CKPT_DIR= os.path.join(os.getcwd(),"training_logs/model_1/checkpoints/")
LOG_DIR = os.getcwd()+ "/logs"
DATASET_DIR = "/media/yongkie/YSDC/dataset_train_rgb_y2/pickles/"
RESULT_DIR = os.path.join(os.getcwd(),"test/")
PRETRAINED_MODEL_10 = "data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl"
PRETRAINED_MODEL_11 = "data/SqueezeNet/squeezenet_v1.1.pkl"

# Image dimensions taken from squeezenet
DEBUG=1
class TLDetector(object):
    def __init__(self):
        #TODO load classifier
        pass

        print('--------------- Loading classifier --------------')
        print
        self.graph = tf.get_default_graph()

        with tf.Session() as self.sess:
        #http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        # Load the meta graph and restore weights
            self.saver = tf.train.import_meta_graph(CKPT_DIR + 'model_1_epoch_3.ckpt.meta')

            print('---------------- Loading complete ---------------')
            model_id = 2
            self.model = SqueezeDet_model(model_id, PRETRAINED_MODEL_11, testmode=True)

            self.model.batch_size = 1
            self.img_height = self.model.img_height
            self.img_width = self.model.img_width
            self.no_of_classes = self.model.no_of_classes


    def calc_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    def resize_image(self, image):
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation = cv2.INTER_LINEAR)
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

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver.restore(self.sess, tf.train.latest_checkpoint(CKPT_DIR))

        #print("graph: ",self.graph.get_operations())
        #print("self.saver: ", self.saver)

        #print("Image: ", image.shape)
        if image is None:
            print("Error: image is None!")
            return
        resized_img = self.resize_image(image)

        if DEBUG:
            print("resized shape: ", resized_img.shape)



        # load the mean color channels of the train imgs:

        with open(DATASET_DIR+"bosch_mean_channels.pkl", "rb") as f:
            train_mean_channels = pickle.load(f, encoding='bytes')
        #preproc_img = helper.preprocess(resized_img, mean_pixel)
        expanded_img = np.expand_dims(resized_img, axis=0) - train_mean_channels
        if DEBUG:
            print('--------------- Getting classification --------------')

        summary_writer = tf.summary.FileWriter(LOG_DIR, graph=self.sess.graph)
        # Placeholders


        # Run prediction
        pred_bboxes, detection_classes, detection_probs  = self.sess.run([self.model.pred_bboxes,
                    self.model.detection_classes, self.model.detection_probs],
                    feed_dict={self.model.imgs_ph:expanded_img,
                               self.model.keep_prob_ph: 1.0})

        # Filter low prediction
        final_bboxes, final_probs, final_classes = self.model.filter_prediction(
                    pred_bboxes[0], detection_probs[0], detection_classes[0])

        print("Final boxes: ", final_probs)
        keep_idx = [idx for idx in range(len(final_probs)) if final_probs[idx] > 0]#plot_prob_thresh
        final_bboxes = [final_bboxes[idx] for idx in keep_idx]
        print("Final boxes: ", final_bboxes)
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_classes = [final_classes[idx] for idx in keep_idx]

        # draw the bboxes outputed by the model:
        pred_img = draw_bboxes(resized_img, final_bboxes, final_classes, final_probs)

        img_name = image_path.split("/")[-1]
        pred_path = RESULT_DIR + img_name.split(".png")[0] + "_pred.png"
        cv2.imwrite(pred_path, pred_img)

@click.command()
@click.argument('img_path')
def main(img_path):
    tlc = TLDetector()
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if os.path.isdir(img_path):
        imgfiles = img_path +"/*.jpg"
        print("test0 ", imgfiles)
        for img_file in sorted(glob.glob(imgfiles)):
            print("test1")
            tlc.get_classification(img_file, 0)
    else:
        print("test2")
        tlc.get_classification(img_path)

if __name__ == '__main__':
    main()