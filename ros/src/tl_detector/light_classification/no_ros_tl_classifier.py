#from styx_msgs.msg import TrafficLight
import tensorflow as tf
import helper
import cv2
import os
import numpy as np
import click
# Get the model directory
MODEL_DIR = os.getcwd()+ "/model"
LOG_DIR = os.getcwd()+ "/logs"

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



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction


        self.saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_DIR))
        #print("graph: ",self.graph.get_operations())
        #print("self.saver: ", self.saver)

        #print("Image: ", image.shape)
        resized_img = helper.resize_image(image)
        print("resized shape: ", resized_img.shape)



        mean_pixel = np.array([104.006, 116.669, 122.679], dtype=np.float32)
        #preproc_img = helper.preprocess(resized_img, mean_pixel)
        expanded_img = np.expand_dims(resized_img, axis=0)
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

        print("Predictions: ", predictions)
        predictions = np.squeeze(predictions)   #squeeze array to 1 dim array
        softmax = helper.calc_softmax(predictions)
        max_index = np.argmax(softmax)
        print("Softmax: ", softmax)
        print("argmax: ", max_index)
        return max_index
        print('--------------- Classification complete -------------')
        #return TrafficLight.UNKNOWN

@click.command()
@click.argument('img_path')
def main(img_path):
    tlc = TLClassifier()
    img = cv2.imread(img_path)
    tlc.get_classification(img)

if __name__ == '__main__':
    main()