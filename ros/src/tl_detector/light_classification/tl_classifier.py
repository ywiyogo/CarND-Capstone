from styx_msgs.msg import TrafficLight
import tensorflow as tf
import helper

import os

# Get the model directory
MODEL_DIR = os.getcwd()+ "/light_classification/model"


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

        print('--------------- Loading classifier --------------')
        print
        sess = tf.Session()
        print
        # Load the meta graph and restore weights
        saver = tf.train.import_meta_graph(MODEL_DIR + '/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
        print('---------------- Loading complete ---------------')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

        print('--------------- Getting classification --------------')

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

            # Get probability in range[0:1] of classification
            softmax_operation = tf.nn.softmax(logits)

            test_probability = sess.run(softmax_operation, feed_dict=
                                            {images: image,
                                            keep_prob: 1})
            test_prediction = np.argmax(test_probability, 1)
            # top_k predictions
            topk = sess.run(tf.nn.top_k(tf.constant(test_probability), k=3))
            print('test_probability', test_probability)
            print('test_prediction', test_prediction)
            print('topk', topk)



        print('--------------- Classification complete -------------')
        return TrafficLight.UNKNOWN
