# Copyright (c) 2017 Andrey Voroshilov

#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
import scipy.io
import time
import helper

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

    x = tf.cast(input_image, get_dtype_tf())

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

    data_dir_LARA = 'data/LARA_dataset/'

    # Hyperparameters
    learning_rate = 1e-4
    epochs = 2
    batch_size = 64
    kp = 0.5

    # Download LARA dataset if not already downloaded
#    helper.maybe_download_LARA_dataset(data_dir_LARA)

    # Load training data generator
    # To run generator: gen = get_batches_fn(batch_size)
    get_batches_fn = helper.gen_batch_function_LARA(data_dir_LARA)

    # Test generator image and label
#    gen = get_batches_fn(batch_size)
#    for image, label in gen:
#	print("testing generator image and label")

    # Loading image
    img_content, orig_shape = imread_resize(options.input)
    img_content_shape = (batch_size,) + img_content.shape

    # Loading ImageNet classes info
    classes = []
    classes.append('red')
    classes.append('yellow')
    classes.append('green')
    classes.append('unknown')

    # Loading network
    data, sqz_mean = load_net('./SqueezeNet/sqz_full.mat')

    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    g = tf.Graph()

    # Simple classification
    with g.as_default(), tf.Session(config=config) as sess:

	# Placeholders
	correct_label = tf.placeholder(dtype=tf.string)
	image         = tf.placeholder(dtype=get_dtype_tf(), shape=img_content_shape, name="image_placeholder")
	keep_prob     = tf.placeholder(get_dtype_tf())

	# SqueezeNet model
	sqznet, logits = net_preloaded(data, image, 'max', True, keep_prob)

	# Loss and Training operations
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
	training_operation = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)


	# Initialize variables
	sess.run(tf.global_variables_initializer())


###<--NEW_start-->###
	print
	print('Training...')
	for epoch in range(epochs):
	    gen = get_batches_fn(batch_size)
	    for images, labels in gen:
		print('images {}, correct_label {}, keep_prob {}, learning_rate {}'.format(images.dtype, labels.dtype, kp.dtype, learning_rate.dtype))

		_, loss = sess.run([training_operation, cross_entropy_loss],
                                   feed_dict={image: images,
                                              correct_label: labels,
                                              keep_prob: kp,
                                              learning_rate: learning_rate}
                                  )
		print('Epoch {}: loss = {}'.format(epoch, loss))
###<--NEW_end-->###


        # Classifying
#        sqznet_results = sqznet['classifier_actv'].eval(feed_dict={image: [preprocess(img_content, sqz_mean)], keep_prob: 1.})[0]

        # Outputting result
#        sqz_class = np.argmax(sqznet_results)

#        print("\nclass: [%d] '%s' with %5.2f%% confidence" % (sqz_class, classes[sqz_class], sqznet_results[sqz_class] * 100))

        
if __name__ == '__main__':
    main()
