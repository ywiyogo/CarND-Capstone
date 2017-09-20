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
        print("graph: ",self.graph.get_operations())
        print("self.saver: ", self.saver)


        print("############################")

        print("Image: ", image.shape)
        resized_img = helper.resize_image(image)
        print("resized shape: ", resized_img.shape)

        expanded_img = np.expand_dims(resized_img, axis=0)

        mean_pixel = np.array([104.006, 116.669, 122.679], dtype=np.float32)
        #preproc_img = self.preprocess(resized_img, mean_pixel)
        print('--------------- Getting classification --------------')
        # Placeholders
        relu_op = self.graph.get_tensor_by_name('Classifier/Relu_2:0')
        summary_writer = tf.summary.FileWriter(LOG_DIR, graph=self.sess.graph)
        #print("graph: ",self.graph.get_operations())
        #     # Get probability in range[0:1] of classification
        predictions = self.sess.run(relu_op, feed_dict=
                                        {"input_images:0": expanded_img,
                                         "keep_prob:0": 0.9})
        predictions = np.squeeze(predictions)
        print("Predictions: ", predictions)
        # sqznet_results = sqznet['classifier_actv'].eval(feed_dict={image: [preprocess(img_content, sqz_mean)], keep_prob: 1.})[0][0][0]
        # sqz_class = np.argmax(sqznet_results)
        # print("sqz class: ", sqz_class)

        # Creates lookup
        # shows the 3 top prediction
        top_3 = predictions.argsort()[-3:][::-1]

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
