from styx_msgs.msg import TrafficLight
from scipy.misc import imresize
import tensorflow as tf
import os
import numpy as np

# Get the model directory
MODEL_DIR = os.getcwd()+ "/light_classification/model"

WIDTH  = 227
HEIGHT = 227


class TLClassifier(object):
    def __init__(self):
        print('--------------- Loading classifier --------------')
        self.graph = tf.get_default_graph()

        self.sess = tf.Session()
        #http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        # Load the meta graph and restore weights
        self.saver = tf.train.import_meta_graph(MODEL_DIR + '/model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_DIR))

        print('---------------- Loading complete ---------------')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        def resize_image(image):
            image = imresize(image, (WIDTH, HEIGHT))
            return image

        def calc_softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x)/np.sum(np.exp(x), axis=0)

        #def preprocess(image, mean_pixel):
            #swap_img = np.array(image)
            #img_out = np.array(swap_img)
            #img_out[:, :, 0] = swap_img[:, :, 2]
            #img_out[:, :, 2] = swap_img[:, :, 0]
            #return img_out - mean_pixel

        #TODO implement light color prediction
        
        #self.saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_DIR))
        #print("Image: ", image.shape)
        resized_img = resize_image(image)
        #print("resized shape: ", resized_img.shape)

        expanded_img = np.expand_dims(resized_img, axis=0)

        #mean_pixel = np.array([104.006, 116.669, 122.679], dtype=np.float32)
        #preproc_img = preprocess(resized_img, mean_pixel)

        # Placeholders
        relu_op = self.graph.get_tensor_by_name('Classifier/Relu_2:0')

        predictions = self.sess.run(relu_op, feed_dict=
                                        {"input_images:0": expanded_img,
                                         "keep_prob:0": 1.})

        #print("Predictions: ", predictions)
        predictions = np.squeeze(predictions)   #squeeze array to 1 dim array

        #Check if all prediction the same or not activated
        if all(val==predictions[0] for val in predictions):
            return TrafficLight.UNKNOWN

        softmax = calc_softmax(predictions)
        max_index = np.argmax(softmax)
        #print("Softmax: ", softmax)

        #print('--------------- Classification {} -------------'.format(max_index))
        if max_index == 0:
            return TrafficLight.RED
        elif max_index == 1:
            return TrafficLight.YELLOW
        elif max_index == 2:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN