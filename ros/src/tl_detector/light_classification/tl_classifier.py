from styx_msgs.msg import TrafficLight
import tensorflow as tf
import helper
import cv2
import os

# Get the model directory
MODEL_DIR = os.getcwd()+ "/light_classification/model"


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

        print('--------------- Loading classifier --------------')
        print
        with tf.Session() as self.sess:
        #http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
        # Load the meta graph and restore weights
            self.saver = tf.train.import_meta_graph(MODEL_DIR + '/model.meta')

            print('---------------- Loading complete ---------------')

    def imread_resize(self, image_orig):
        #img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
        img = cv2.resize(image_orig,(227,227))
        if len(img.shape) == 2:
            # grayscale
            img = np.dstack((img,img,img))
        return img, img.shape

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction


        self.saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_DIR))
        print("self.saver: ", self.saver)

        self.graph = tf.get_default_graph()

        print("Image: ", image.shape)
        resized_img, imgshape = self.imread_resize(image)
        print("resized shape: ", imgshape)
        print('--------------- Getting classification --------------')
        # Placeholders
        # logits = tf.placeholder(dtype=tf.float32, name="logits")
        # softmax_operation = tf.nn.softmax(logits)
        # #     # Get probability in range[0:1] of classification

        # predictions = self.sess.run(softmax_operation, feed_dict=
        #                                 {'input_images:0': resized_img})
        # predictions = np.squeeze(predictions)
        # print("Predictions: ", predictions)
        # # Creates lookup
        # # shows the 3 top prediction
        # top_3 = predictions.argsort()[-3:][::-1]

        #test_prediction = np.argmax(test_probability, 1)
        #     # top_k predictions
        #     topk = sess.run(tf.nn.top_k(tf.constant(test_probability), k=3))
        #     print('test_probability', test_probability)
        #     print('test_prediction', test_prediction)
        #     print('topk', topk)

# Classifying
        # sqznet_results = self.saver['classifier_actv'].eval(feed_dict={image: [preprocess(image, sqz_mean)], keep_prob: 1.})[0]

        # # Outputting result
        # sqz_class = np.argmax(sqznet_results)

        # print("\nclass: [%d] '%s' with %5.2f%% confidence" % (sqz_class, classes[sqz_class], sqznet_results[sqz_class] * 100))

        print('--------------- Classification complete -------------')
        return TrafficLight.UNKNOWN
