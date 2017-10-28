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

        print('--------------- Loading TF Graph --------------')
        print
        self.graph = tf.get_default_graph()

        model_id = 2
        self.model = SqueezeDet_model(model_id, pretrained_model_path=None, testmode=True)
        print("Batch size: ", self.model.batch_size)

        self.saver = tf.train.Saver(self.model.model_params)

        self.sess = tf.Session(config=tf.ConfigProto())
        #http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        # Load the meta graph and restore weights

        print('---------------- Loading Checkpoint model ---------------')

        self.saver.restore(self.sess, CKPT_DIR + 'model_1_epoch_4.ckpt')

        with open(DATASET_DIR+"bosch_mean_channels.pkl", "rb") as f:
            self.train_mean_channels = pickle.load(f, encoding='bytes')

    def calc_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    def resize_image(self, image):
        image = cv2.resize(image, (self.model.img_width, self.model.img_height), interpolation = cv2.INTER_LINEAR)
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

        #print("graph: ",self.graph.get_operations())
        #print("self.saver: ", self.saver)

        #print("Image: ", image.shape)
        if image is None:
            print("Error: image is None!")
            return
        image = image.astype(np.float32, copy=False)
        resized_img = cv2.resize(image, (self.model.img_width, self.model.img_height))

        if DEBUG:
            print("resized shape: ", resized_img.shape)

        #preproc_img = helper.preprocess(resized_img, mean_pixel)
        expanded_img = np.expand_dims(resized_img, axis=0) - self.train_mean_channels

        #relu_op = self.graph.get_tensor_by_name('InterpretOutput/pred_confidence_score:0')
        summary_writer = tf.summary.FileWriter(LOG_DIR, graph=self.sess.graph)
        # Placeholders
        print("model input placholder: ", self.model.image_input)
        print("input: ", expanded_img.shape)

        # Run prediction
        #a = self.sess.run(relu_op,feed_dict={"imgs_ph:0":expanded_img,
        #                                     "keep_prob_ph:0": 1.})
        # Detect
        det_boxes, det_probs, det_class = self.sess.run(
            [self.model.pred_bboxes, self.model.detection_probs, self.model.detection_classes],
            feed_dict={self.model.image_input:expanded_img,
                       self.model.keep_prob_ph: 1.0})
        self.sess.close()


        # Filter low prediction
        final_bboxes, final_probs, final_classes = self.model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

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