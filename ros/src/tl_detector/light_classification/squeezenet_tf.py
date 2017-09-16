# Copyright (c) 2017 Andrey Voroshilov

#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
import scipy.io
import time
import helper
import cv2

from tensorflow.contrib.layers import flatten

from PIL import Image

from argparse import ArgumentParser

def imread_resize(path):
    img_orig = scipy.misc.imread(path)
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img, img_orig.shape

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
    
def get_dtype_np():
    return np.float32

def get_dtype_tf():
    return tf.float32
    
# SqueezeNet v1.1 (signature pool 1/3/5)
########################################

def load_net(data_path):
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)

    # Converting to needed type
    conv_time = time.time()
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append( kernels.astype(get_dtype_np()) )
            weights[name].append( bias.astype(get_dtype_np()) )
    print("Converted network data(%s): %fs" % (get_dtype_np(), time.time() - conv_time))
    
    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    return weights, mean_pixel

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def unprocess(image, mean_pixel):
    swap_img = np.array(image + mean_pixel)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out

def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

def fire_cluster(net, x, preloaded, cluster_name):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x = _act_layer(net, layer_name + '_actv', x)
    
    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_l = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_r = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='SAME')
    x_r = _act_layer(net, layer_name + '_actv', x_r)
    
    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x
    
    return x

def net_preloaded(preloaded, input_image, pooling, needs_classifier=False, keep_prob=None):
    net = {}
    cr_time = time.time()

#    x = tf.cast(input_image, get_dtype_tf())
    x = input_image

    # Feature extractor
    #####################
    
    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
    x = _act_layer(net, layer_name + '_actv', x)
    x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire2 + fire3 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire2')
    fire2_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire3')
    x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire4 + fire5 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire4')
    fire4_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire5')
    x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # remainder (no pooling)
    x = fire_cluster(net, x, preloaded, cluster_name='fire6')
    fire6_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire7')
    x = fire_cluster(net, x, preloaded, cluster_name='fire8')
    x = fire_cluster(net, x, preloaded, cluster_name='fire9')
    
    # Classifier
    #####################
    if needs_classifier == True:
        # Dropout [use value of 50% when training]
        x = tf.nn.dropout(x, keep_prob)
    
        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases)
        x = _act_layer(net, layer_name + '_actv', x)
        
        # Global Average Pooling
        x = _pool_layer(net, 'classifier_pool', x, 'avg', size=(13, 13), stride=(1, 1), padding='VALID')

        # Flatten. Input = 1x1x1x1000. Output = 1x1000.
        fc0 = flatten(x)

        # Fully Connected. Input = 1000. Output = 100.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(1000, 100), mean = 0, stddev = 0.1))
        fc1_b = tf.Variable(tf.zeros(100))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        # Activation.
        fc1 = tf.nn.relu(fc1)

        # Fully Connected. Input = 100. Output = 20.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 20), mean = 0, stddev = 0.1))
        fc2_b = tf.Variable(tf.zeros(20))
        fc2   = tf.matmul(fc1, fc2_W) + fc2_b
        # Activation.
        fc2 = tf.nn.relu(fc2)

        # Fully Connected. Input = 20. Output = 4.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(20, 4), mean = 0, stddev = 0.1))
        fc3_b = tf.Variable(tf.zeros(4))
        fc3   = tf.matmul(fc2, fc3_W) + fc3_b
        # Activation.
        logits = tf.nn.relu(fc3)

        net['classifier_actv'] = logits

    print("Network instance created: %fs" % (time.time() - cr_time))
   
    return net, logits
    
def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
            padding=padding)
    x = tf.nn.bias_add(conv, bias)
    net[name] = x
    return x

def _act_layer(net, name, input):
    x = tf.nn.relu(input)
    net[name] = x
    return x
    
def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    net[name] = x
    return x

def build_parser():
    ps = ArgumentParser()
    ps.add_argument('--in', dest='input', help='input file', metavar='INPUT', required=True)
    return ps

def main():

    parser = build_parser()
    options = parser.parse_args()

    # Loading network
    data, sqz_mean = load_net('./SqueezeNet/sqz_full.mat')

    # Hyperparameters
    lr = 1e-4
    epochs = 1
    batch_size = 128
    kp = 0.5

    # Load training data generator
    data_dir_LARA = 'data/LARA_dataset/'
    get_batches_fn, X_test, y_test = helper.gen_batch_function_LARA(data_dir_LARA)

    # Test generator image and label
#    gen = get_batches_fn(batch_size)
#    for image, label in gen:
#	print("testing generator image and label")

    # Placeholders
    images        = tf.placeholder(dtype=tf.float32, shape=(batch_size, helper.HEIGHT, helper.WIDTH, 3))
    labels        = tf.placeholder(dtype=tf.int32, shape=batch_size)
    keep_prob     = tf.placeholder(dtype=tf.float32)
    learning_rate = tf.placeholder(dtype=tf.float32)

    # SqueezeNet model
    model, logits = net_preloaded(data, images, 'max', True, keep_prob)

    # Loss and Training operations
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    training_operation = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

    # Accuracy operation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Evaluate the loss and accuracy of the model
    def evaluate(X_data, y_data, batch_size):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
            accuracy = sess.run(accuracy_operation, feed_dict={images: batch_x,
                                                               labels: batch_y,
                                                               keep_prob: 1})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    # Save variables
    saver = tf.train.Saver()

    # Loading image
    image = helper.get_image(options.input)

    '''
    Load Traffic Light Classifier classes
    0 = RED
    1 = YELLOW
    2 = GREEN
    4 = UNKNOWN
    '''
    classes = [0, 1, 2, 4]
    num_classes = len(classes)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    # Simple classification
    with tf.Session(config=config) as sess:

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        print
        print('Training...')
        for epoch in range(epochs):
            gen = get_batches_fn(batch_size)
            for X_train, y_train in gen:

                _, loss = sess.run([training_operation, cross_entropy_loss],
                                   feed_dict={images: X_train,
                                              labels: y_train,
                                              keep_prob: kp,
                                              learning_rate: lr}
                                  )
                print('Epoch {}: loss = {}'.format(epoch+1, loss))

        # Save the variables to disk.
        saver.save(sess, "model/model")
        print("Model saved.")


        test_accuracy = evaluate(X_test, y_test, batch_size)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

        '''
        # serialize model to YAML
        print('Saving model as model.yaml...')
        with open("model.yaml", "w") as outfile:
            yaml.dump(model, outfile, default_flow_style=False)
        # serialize weights to HDF5
        hf = h5py.File('model.h5', 'w')
        hf.create_dataset('model', data=)
        print('Saving weights as model.h5...')
        model.save_weights("model.h5")
        print("Saved model and weights to disk")
		'''

        '''
        # Classifying
#        sqznet_results = model['classifier_actv'].eval(feed_dict={image: [preprocess(image, sqz_mean)], keep_prob: 1.})[0]

        # Outputting result
#        sqz_class = np.argmax(sqznet_results)

#        print("\nclass: [%d] '%s' with %5.2f%% confidence" % (sqz_class, classes[sqz_class], sqznet_results[sqz_class] * 100))
         '''
        
if __name__ == '__main__':
    main()
