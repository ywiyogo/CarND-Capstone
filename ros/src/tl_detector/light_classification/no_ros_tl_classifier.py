#from styx_msgs.msg import TrafficLight
import tensorflow as tf
import helper
import cv2
import os
import numpy as np
# Get the model directory
MODEL_DIR = os.getcwd()+ "/model"


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

    def imread_resize(self, image_orig):
        #img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
        img = cv2.resize(image_orig,(227,227))
        imgs = np.empty((helper.batch_size, 227, 227, 3))
        for k in xrange(helper.batch_size):
            imgs[k,:,:,:] = img
        if len(img.shape) == 2:
            # grayscale
            img = np.dstack((img,img,img))
        new_shape = (helper.batch_size,) + img.shape
        return imgs, new_shape

    def preprocess(self, image, mean_pixel):
        swap_img = np.array(image)
        img_out = np.array(swap_img)
        img_out[:, :, 0] = swap_img[:, :, 2]
        img_out[:, :, 2] = swap_img[:, :, 0]
        return img_out - mean_pixel

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
        resized_img, resiz_imgshape = self.imread_resize(image)
        mean_pixel = np.array([104.006, 116.669, 122.679], dtype=np.float32)
        #preproc_img = self.preprocess(resized_img, mean_pixel)
        print("resized shape: ", resiz_imgshape)
        print('--------------- Getting classification --------------')
        # Placeholders
        logits = tf.placeholder(dtype=tf.float32)
        softmax_operation = tf.nn.softmax(logits)
        print("graph: ",self.graph.get_operations())
        #     # Get probability in range[0:1] of classification
        predictions = self.sess.run(softmax_operation, feed_dict=
                                        {"input_images:0": resized_img,
                                        "Placeholder_2:0": 0.01,
                                        "Placeholder_3:0": 0.8})
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
        #return TrafficLight.UNKNOWN

if __name__ == '__main__':
    tlc = TLClassifier()
    img = cv2.imread(os.getcwd()+ "/images/Sim_images/green.jpg")
    print(img.shape)
    tlc.get_classification(img)