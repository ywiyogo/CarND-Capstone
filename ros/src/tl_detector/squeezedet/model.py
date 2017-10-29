# Implementation of the SqueezeDet model
# Note: This code is based on https://github.com/fregu856/2D_detection and
#  github.com/BichenWuUCB/squeezeDet

import numpy as np
import tensorflow as tf
import os
import joblib   # like pickle but it seems to be faster

from utilities import safe_exp, bbox_transform, bbox_transform_inv, nms

def analyse_tf_graph(MODEL_PATH):
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(cw)

    #sess.run(tf.global_variables_initializer())
    model_dir = os.path.join(os.getcwd(),"data","pretrainedmodel")
    saver.save(sess, model_dir)

    saver = tf.train.import_meta_graph(model_dir+".meta")
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    layer1 = graph.get_tensor_by_name("conv1")
    print(layer1)

def _add_loss_summaries(total_loss):
    """Add summaries for losses
    Generates loss summaries for visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    """
    losses = tf.get_collection('losses')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)

class SqueezeDet_model(object):

    def __init__(self, model_id, pretrained_model_path=None, testmode=False):
        self.model_id = model_id

        self.project_dir = project_dir = os.getcwd()
        self.logs_dir =  os.path.join(self.project_dir, "training_logs/")
        if testmode:
            self.batch_size = 1
        else:
            self.batch_size = 20

        self.img_height = 720
        self.img_width = 1280

        self.num_classes = 4
        self.class_string_to_label = {"Red": 0, "Yellow": 1, "Green": 2,
                                      "off": 3}
        # capacity for FIFOQueue
        # self.queue_capacity = 2
        # model parameters
        self.model_params = []

        # model size counter
        self.model_size_counter = [] # array of tuple of layer name, parameter size
        # flop counter
        self.flop_counter = [] # array of tuple of layer name, flop number

        # training parameters:
        self.initial_lr = 0.0001
        self.decay_steps =  1000
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.max_grad_norm = 1.0
        self.weight_decay = 0.0001

        # set all anchor bboxes and related parameters:
        self.anchor_bboxes = self.set_anchors()
        # # (anchor_bboxes has shape [anchors_per_img, 4])
        self.anchors_per_img = len(self.anchor_bboxes)
        self.anchors_per_gridpoint = 9

        # bbox filtering parameters for testing:
        self.top_N_detections = 10
        self.prob_thresh = 0.2
        self.nms_thresh = 0.4
        self.plot_prob_thresh = 0.6

        # other general parameters:
        self.exp_thresh = 1.0
        self.epsilon = 1e-16

        # loss coefficients:
        self.loss_coeff_class = 0.5
        self.loss_coeff_conf_pos = 75.0
        self.loss_coeff_conf_neg = 100.0
        self.loss_coeff_bbox = 1.0

        self.load_pretrained_model = False

        if not pretrained_model_path is None:
            self.load_pretrained_model = True
            # get the weights of a pretrained SqueezeNet model:
            model_path = os.path.join(os.getcwd(),pretrained_model_path)
            assert os.path.exists(model_path), "Cannot find retrained model: %s " % model_path
            self.caffemodel_weights = joblib.load(model_path)


        # create all dirs for storing checkpoints and other log data:
        self.create_model_dirs()

        # add placeholders to the comp. graph:
        self.add_placeholders()

        # add everything related to the model forward pass to the comp. graph:
        self.add_forward_pass()

        # process the model output and add to the comp. graph:
        self.add_processed_output()

        # compute the batch loss and add to the comp. graph:
        self.add_loss_op()

        if not testmode:
            # add a training operation (for minimizing the loss) to the comp. graph:
            self.add_train_op()

    def create_model_dirs(self):
        self.model_dir = self.logs_dir + "model_%s" % self.model_id + "/"
        self.checkpoints_dir = self.model_dir + "checkpoints/"
        self.debug_imgs_dir = self.model_dir + "imgs/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.debug_imgs_dir)

    def add_placeholders(self):
        with tf.variable_scope('Inputs') as scope:
            self.image_input_ph = tf.placeholder(tf.float32,
                        shape=[self.batch_size, self.img_height, self.img_width, 3],
                        name="image_input")

            self.keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob_ph")

            self.box_mask_ph = tf.placeholder(tf.float32,
                        shape=[self.batch_size, self.anchors_per_img, 1],
                        name='box_mask')
            # # (mask_ph[i, j] == 1 if anchor j is assigned to (i.e., is responsible
            # # for detecting) a ground truth bbox in batch image i, 0 otherwise)

            self.box_delta_input_ph = tf.placeholder(tf.float32,
                        shape=[self.batch_size, self.anchors_per_img, 4],
                        name='box_delta_input')
            # # (if anchor j is assigned to a ground truth bbox in batch image i,
            # # gt_deltas_ph[i, j] == [delta_x, delta_y, delta_w, delta_h] where the
            # # deltas transform anchor j into its assigned ground truth bbox via
            # # eq. (1) in the paper. Otherwise, gt_deltas_ph[i, j] == [0, 0, 0, 0])

            self.gt_boxes_input_ph = tf.placeholder(tf.float32,
                        shape=[self.batch_size, self.anchors_per_img, 4],
                        name='gt_bbox_input')
            # # (if anchor j is assigned to a ground truth bbox in batch image i,
            # # gt_bboxes_ph[i, j] == [center_x, center_y, w, h] of this assigned
            # # ground truth bbox. Otherwise, gt_bboxes_ph[i, j] == [0, 0, 0, 0])

            self.class_labels_ph = tf.placeholder(tf.float32,
                        shape=[self.batch_size, self.anchors_per_img,
                        self.num_classes], name="class_labels")
            # # (if anchor j is assigned to a ground truth bbox in batch image i,
            # # class_labels_ph[i, j] is the onehot encoded class label of this
            # # assigned ground truth bbox. Otherwise, class_labels_ph[i, j] is all
            # # zeros)

        self.ious = tf.Variable(
            initial_value=np.zeros((self.batch_size, self.anchors_per_img)), trainable=False,
            name='iou', dtype=tf.float32
        )
            # FIFOQueue and Batch causes an explosion of RAM consuption
            # self.FIFOQueue = tf.FIFOQueue(
            #     capacity=self.queue_capacity,
            #     dtypes=[tf.float32, tf.float32, tf.float32,
            #             tf.float32, tf.float32],
            #     shapes=[[self.img_height, self.img_width, 3],
            #             [self.anchors_per_img, 1],
            #             [self.anchors_per_img, 4],
            #             [self.anchors_per_img, 4],
            #             [self.anchors_per_img, self.num_classes]],
            # )

            # self.enqueue_op = self.FIFOQueue.enqueue_many(
            #     [self.image_input_ph, self.box_mask_ph, self.box_delta_input_ph, self.gt_boxes_input_ph, self.class_labels_ph]
            # )

        # Variable with values
        # self.image_input, self.box_mask, self.box_delta_input, self.gt_boxes_input, self.class_labels = tf.train.batch(
        #         self.FIFOQueue.dequeue(), batch_size=self.batch_size,
        #         capacity=self.queue_capacity)

    def create_feed_dict(self, imgs, keep_prob, mask=None, gt_deltas=None,
                         gt_bboxes=None, class_labels=None):
        # return a feed_dict mapping the placeholders to the actual input data:
        feed_dict = {}
        feed_dict[self.image_input_ph] = imgs
        feed_dict[self.keep_prob_ph] = keep_prob
        if mask is not None: # (we have no mask during inference)
            feed_dict[self.box_mask_ph] = mask
        if gt_deltas is not None:
            feed_dict[self.box_delta_input_ph] = gt_deltas
        if gt_bboxes is not None:
            feed_dict[self.gt_boxes_input_ph] = gt_bboxes
        if class_labels is not None:
            feed_dict[self.class_labels_ph] = class_labels

        return feed_dict

    def add_forward_pass(self):
        # (NOTE! the layer names ("conv1", "fire2" etc.) below must match
        # the names in the pretrained SqueezeNet model when using this for
        # initialization)
        conv_1 = self.conv_layer("conv1", self.image_input_ph, filters=64, size=3,
                    stride=2, padding="SAME", freeze=True)
        pool_1 = self.pooling_layer(conv_1, size=3, stride=2, padding="SAME")

        fire_2 = self.fire_layer("fire2", pool_1, s1x1=16, e1x1=64, e3x3=64)
        fire_3 = self.fire_layer("fire3", fire_2, s1x1=16, e1x1=64, e3x3=64)
        pool_3 = self.pooling_layer(fire_3, size=3, stride=2, padding="SAME")

        fire_4 = self.fire_layer("fire4", pool_3, s1x1=32, e1x1=128, e3x3=128)
        fire_5 = self.fire_layer("fire5", fire_4, s1x1=32, e1x1=128, e3x3=128)
        pool_5 = self.pooling_layer(fire_5, size=3, stride=2, padding="SAME")

        fire_6 = self.fire_layer("fire6", pool_5, s1x1=48, e1x1=192, e3x3=192)
        fire_7 = self.fire_layer("fire7", fire_6, s1x1=48, e1x1=192, e3x3=192)
        fire_8 = self.fire_layer("fire8", fire_7, s1x1=64, e1x1=256, e3x3=256)
        fire_9 = self.fire_layer("fire9", fire_8, s1x1=64, e1x1=256, e3x3=256)

        fire_10 = self.fire_layer("fire10", fire_9, s1x1=96, e1x1=384, e3x3=384)
        fire_11 = self.fire_layer("fire11", fire_10, s1x1=96, e1x1=384, e3x3=384)
        dropout_11 = tf.nn.dropout(fire_11, self.keep_prob_ph, name="dropout_11")

        # see the paper: K(4+1+C)
        no_of_outputs = self.anchors_per_gridpoint*(self.num_classes + 1 + 4)

        self.preds = self.conv_layer("conv12", dropout_11, filters=no_of_outputs,
                    size=3, stride=1, padding="SAME", relu=False, stddev=0.0001)

    def add_processed_output(self):
        preds = self.preds

        # get all predicted class probabilities:
        # # compute the total number of predicted class probs per grid point:
        with tf.variable_scope('InterpretOutput') as scope:
            no_of_class_probs = self.anchors_per_gridpoint * self.num_classes
            print("no_of_class_probs:", no_of_class_probs)
            # # get all predicted class logits:
            pred_class_logits = preds[:, :, :, :no_of_class_probs]
            print("pred_class_logits:", pred_class_logits)

            pred_class_logits = tf.reshape(pred_class_logits,
                        [-1, self.num_classes])
            # # convert the class logits to class probs:
            pred_class_probs = tf.nn.softmax(pred_class_logits, name='pred_class_probs')

            print("no_of_class_probs:", no_of_class_probs)
            print("pred_class_probs:", pred_class_probs)
            print("array:", [self.batch_size, self.anchors_per_img, self.num_classes])

            pred_class_probs = tf.reshape(pred_class_probs,
                        [self.batch_size, self.anchors_per_img, self.num_classes])
            self.pred_class_probs = pred_class_probs

            # get all predicted confidence scores:
            # # compute the total number of predicted conf scores per grid point:
            no_of_conf_scores = self.anchors_per_gridpoint
            # # get all predicted conf scores:
            pred_conf_scores = preds[:, :, :, no_of_class_probs:no_of_class_probs + no_of_conf_scores]
            pred_conf_scores = tf.reshape(pred_conf_scores,
                        [self.batch_size, self.anchors_per_img])
            # # normalize the conf scores to lay between 0 and 1:
            pred_conf_scores = tf.sigmoid(pred_conf_scores, name='pred_confidence_score')
            self.pred_conf_scores = pred_conf_scores

            # get all predicted bbox deltas (the four numbers that describe how to
            # transform an anchor bbox to a predicted bbox):
            pred_bbox_deltas = preds[:, :, :, no_of_class_probs + no_of_conf_scores:]

            pred_bbox_deltas = tf.reshape(pred_bbox_deltas,
                            [self.batch_size, self.anchors_per_img, 4], name='bbox_delta')
            self.pred_bbox_deltas = pred_bbox_deltas

            # compute the total number of ground truth objects in the batch (used to
            # normalize the bbox and classification losses):
            self.no_of_gt_objects = tf.reduce_sum(self.box_mask_ph, name='num_objects')

        with tf.variable_scope("BBox") as scope:
            # transform the anchor bboxes to predicted bboxes using the predicted
            # bbox deltas:
            delta_x, delta_y, delta_w, delta_h = tf.unstack(self.pred_bbox_deltas, axis=2)
            # # (delta_x has shape [batch_size, anchors_per_img])
            anchor_x = self.anchor_bboxes[:, 0]
            anchor_y = self.anchor_bboxes[:, 1]
            anchor_w = self.anchor_bboxes[:, 2]
            anchor_h = self.anchor_bboxes[:, 3]
            # # transformation according to eq. (1) in the paper:
            bbox_center_x = anchor_x + anchor_w*delta_x
            bbox_center_y = anchor_y + anchor_h*delta_y
            bbox_width = anchor_w*safe_exp(delta_w, self.exp_thresh)
            bbox_height = anchor_h*safe_exp(delta_h, self.exp_thresh)

            # # trim the predicted bboxes so that they stay within the image:
            # # # get the max and min x and y coordinates for each predicted bbox
            # # # from the predicted center coordinates and height/width (these
            # # # might lay outside of the image (e.g. be negative or larger than
            # # # the img width)):
            xmin, ymin, xmax, ymax = bbox_transform([bbox_center_x, bbox_center_y,
                        bbox_width, bbox_height])
            # # # limit xmin to be in [0, img_width - 1]:
            xmin = tf.minimum(tf.maximum(0.0, xmin), self.img_width - 1.0)
            # # # limit ymin to be in [0, img_height - 1]:
            ymin = tf.minimum(tf.maximum(0.0, ymin), self.img_height - 1.0)
            # # # limit xmax to be in [0, img_width - 1]:
            xmax = tf.maximum(tf.minimum(self.img_width - 1.0, xmax), 0.0)
            # # # limit ymax to be in [0, img_height - 1]:
            ymax = tf.maximum(tf.minimum(self.img_height - 1.0, ymax), 0.0)
            # # # transform the trimmed bboxes back to center/width/height format:
            cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
            #===================================
            # Used for test operation in demo.py
            #===================================
            self.pred_bboxes = tf.transpose(tf.stack([cx, cy, w, h]), (1, 2, 0), name='pred_bbox')

        with tf.variable_scope("IOU"):
            def _tensor_iou(box1, box2):
                with tf.variable_scope('intersection'):
                  xmin = tf.maximum(box1[0], box2[0], name='xmin')
                  ymin = tf.maximum(box1[1], box2[1], name='ymin')
                  xmax = tf.minimum(box1[2], box2[2], name='xmax')
                  ymax = tf.minimum(box1[3], box2[3], name='ymax')

                  w = tf.maximum(0.0, xmax-xmin, name='inter_w')
                  h = tf.maximum(0.0, ymax-ymin, name='inter_h')
                  intersection = tf.multiply(w, h, name='intersection')

                with tf.variable_scope('union'):
                  w1 = tf.subtract(box1[2], box1[0], name='w1')
                  h1 = tf.subtract(box1[3], box1[1], name='h1')
                  w2 = tf.subtract(box2[2], box2[0], name='w2')
                  h2 = tf.subtract(box2[3], box2[1], name='h2')

                  union = w1*h1 + w2 * h2 - intersection

                return intersection/(union + self.epsilon) \
                    * tf.reshape(self.box_mask_ph, [self.batch_size, self.anchors_per_img])

            self.ious = self.ious.assign(
              _tensor_iou(
                  bbox_transform(tf.unstack(self.pred_bboxes, axis=2)),
                  bbox_transform(tf.unstack(self.gt_boxes_input_ph, axis=2))
              )
            )

        # compute Pr(class) = Pr(class | object)*Pr(object):

        with tf.name_scope("Probability"):

            probs = self.pred_class_probs * tf.reshape(self.pred_conf_scores,
                                                      [self.batch_size, self.anchors_per_img, 1],
                                                      name='final_class_prob')
            #===================================
            # Used for test operation in demo.py
            #===================================
            # for each predicted bbox, compute the predicted probability that it
            # actually contains this most likely object class:
            self.detection_probs = tf.reduce_max(probs, 2, name='score')
            # for each predicted bbox, compute what object class it most likely
            # contains according to the model output:
            self.detection_classes = tf.argmax(probs, 2, name='class_idx')

        # # (self.detection_probs and self.detection_classes have shape
        # # [batch_size, anchors_per_img])

    def add_loss_op(self):
        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up
        with tf.variable_scope('class_regression') as scope:
            class_loss = (self.class_labels_ph *(-tf.log(self.pred_class_probs + self.epsilon)) +
                        (1 - self.class_labels_ph)*(-tf.log(1 - self.pred_class_probs + self.epsilon)))
            class_loss = self.loss_coeff_class*self.box_mask_ph * class_loss
            class_loss = tf.reduce_sum(class_loss, name="sum_class_loss")
            # # normalize the class loss (tf.truediv is used to ensure that we get
            # # no integer divison):
            class_loss = tf.truediv(class_loss, self.no_of_gt_objects, name="class_loss")
            self.class_loss = class_loss
            tf.add_to_collection("losses", self.class_loss)


        # compute the confidence score regression loss (this doesn't look like
        # the conf loss in the paper, but they are actually equivalent since
        # self.IOUs is masked as well):
        with tf.variable_scope('ConfidenceScoreRegression') as scope:
            input_mask = tf.reshape(self.box_mask_ph, [self.batch_size,
                        self.anchors_per_img])

            self.conf_loss = tf.reduce_mean(
                  tf.reduce_sum(
                      tf.square((self.ious - self.pred_conf_scores))
                      * (input_mask*self.loss_coeff_conf_pos/self.no_of_gt_objects
                         +(1-input_mask)*self.loss_coeff_conf_neg/(self.anchors_per_img - self.no_of_gt_objects)),
                      reduction_indices=[1]
                  ),
                  name='confidence_loss'
              )
            tf.add_to_collection('losses', self.conf_loss)
            tf.summary.scalar('mean iou', tf.reduce_sum(self.ious)/self.no_of_gt_objects)
        # # (not sure if we're actually supposed to use reduce_mean in this loss,
        # # (not sure if we're actually supposed to use reduce_mean in this loss,
        # # if so I feel like we should divide with the number of gt objects per
        # # img instead. Think this might be why self.loss_coeff_conf_pos/neg is
        # # so much larger than the other coefficients)

        # compute the bbox regression loss:
        with tf.variable_scope('BBoxRegression') as scope:
            bbox_loss = self.box_mask_ph * (self.pred_bbox_deltas - self.box_delta_input_ph)
            bbox_loss = self.loss_coeff_bbox*tf.square(bbox_loss)
            bbox_loss = tf.reduce_sum(bbox_loss)
            bbox_loss = tf.truediv(bbox_loss, self.no_of_gt_objects, name='bbox_loss')
            self.bbox_loss = bbox_loss
            tf.add_to_collection("losses", self.bbox_loss)

        # compute the total loss by summing the above losses and all variable
        # weight decay losses:
        self.loss = tf.add_n(tf.get_collection("losses"))

    def add_train_op(self):
        # create an optimizer:
        global_step = tf.Variable(0, name="global_step", trainable=False)

        lr = tf.train.exponential_decay(learning_rate=self.initial_lr,
                        global_step=global_step, decay_steps=self.decay_steps,
                        decay_rate=self.lr_decay_rate, staircase=True)

        tf.summary.scalar('learning_rate', lr)
        _add_loss_summaries(self.loss)

        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.momentum)

        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())

        # perform maximum clipping of the gradients:
        with tf.name_scope("clip_gradient"):
            for i, (grad, var) in enumerate(grads_and_vars):
                grads_and_vars[i] = (tf.clip_by_norm(grad, self.max_grad_norm), var)

        # create the train op (global_step will automatically be incremented):
        apply_gradient_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
            self.train_op = tf.no_op(name='train')

    def fire_layer(self, layer_name, input, s1x1, e1x1, e3x3, stddev=0.01, freeze=False):
        # (NOTE! the layer names ("/squeeze1x1" etc.) below must match the
        # names in the pretrained SqueezeNet model when using this for
        # initialization)

        sq1x1 = self.conv_layer(layer_name + "/squeeze1x1", input, filters=s1x1,
                    size=1, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        ex1x1 = self.conv_layer(layer_name + "/expand1x1", sq1x1, filters=e1x1,
                    size=1, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        ex3x3 = self.conv_layer(layer_name + "/expand3x3", sq1x1, filters=e3x3,
                    size=3, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

    def conv_layer(self, layer_name, input, filters, size, stride, padding="SAME",
                   freeze=False, xavier=False, relu=True, stddev=0.001):
        with tf.variable_scope(layer_name) as scope:
            channels = input.get_shape().as_list()[3]

            use_pretrained_params = False
            if self.load_pretrained_model:
                # get the pretrained parameter values if possible:
                cw = self.caffemodel_weights    #type is a dict

                #for key, value in cw.items() :
                    #print(key)

                if layer_name in cw:
                    # re-order the caffe kernel with shape [filters, channels, h, w]
                    # to a tf kernel with shape [h, w, channels, filters]:
                    kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
                    bias_val = cw[layer_name][1]
                    # check the shape:
                    #print("kernel_val: %s compared to %s" %(kernel_val.shape, (size, size, channels, filters)))
                    if kernel_val.shape == (size, size, channels, filters) and (bias_val.shape == (filters, )):
                        use_pretrained_params = True
                        print("kernel shape of %s match!, Use the pretrained parameters"% layer_name)
                    else:
                        print("Shape of the pretrained parameter of %s does not match, use randomly initialized parameter" % layer_name)
                else:
                    print("Cannot find %s in the pretrained model. Use randomly initialized parameters" % layer_name)


            # create the parameter initializers:
            if use_pretrained_params:
                print("Using pretrained init for " + layer_name)

                kernel_init = tf.constant(kernel_val , dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                print("Using Xavier init for " + layer_name)

                kernel_init = tf.contrib.layers.xavier_initializer()
                bias_init = tf.constant_initializer(0.0)
            else:
                print("Using random normal init for " + layer_name)

                kernel_init = tf.truncated_normal_initializer(stddev=stddev,
                            dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            # create the variables:
            kernel = self.variable_with_weight_decay("kernel",
                        shape=[size, size, channels, filters], wd=self.weight_decay,
                        initializer=kernel_init, trainable=(not freeze))
            biases = self.get_variable("biases", shape=[filters], dtype=tf.float32,
                        initializer=bias_init, trainable=(not freeze))
            #Used for testing in demo.py
            self.model_params += [kernel, biases]

            # convolution:
            conv = tf.nn.conv2d(input, kernel, strides=[1, stride, stride, 1],
                        padding=padding, name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            # apply ReLu if supposed to:
            if relu:
                out = tf.nn.relu(conv_bias, "relu")
            else:
                out = conv_bias

            return out

    def pooling_layer(self, input, size, stride, padding="SAME"):
        with tf.name_scope("Pooling"):
            out = tf.nn.max_pool(input, ksize=[1, size, size, 1],
                    strides=[1, stride, stride, 1], padding=padding)

        return out

    def filter_prediction(self, boxes, probs, class_inds):
        # Filter prediction for testing in demo.py
        #(boxes, probs and class_inds are lists of length anchors_per_img)
        print("Probs: ", probs)
        print("length probs: ", len(probs))
        print("Class idx: ", class_inds)
        if self.top_N_detections < len(probs):
            # get the top_N_detections largest probs and their corresponding
            # boxes and class_inds:
            # # (order[0] is the index of the largest value in probs, order[1]
            # # the index of the second largest value etc. order has length
            # # top_N_detections)
            order = probs.argsort()[:-self.top_N_detections-1:-1]
            probs = probs[order]
            boxes = boxes[order]
            class_inds = class_inds[order]
        else:
            # remove all boxes, probs and class_inds corr. to
            # prob values <= prob_thresh:
            filtered_idx = np.nonzero(probs > self.prob_thresh)[0]
            probs = probs[filtered_idx]
            boxes = boxes[filtered_idx]
            class_inds = class_inds[filtered_idx]
        print("Filter threshold: ", self.prob_thresh)
        print("Probabilities after filter: ", probs)
        final_boxes = []
        final_probs = []
        final_class_inds = []
        for c in range(self.num_classes):
            inds_for_c = [i for i in range(len(probs)) if class_inds[i] == c]
            keep = nms(boxes[inds_for_c], probs[inds_for_c], self.nms_thresh)
            for i in range(len(keep)):
                if keep[i]:
                  final_boxes.append(boxes[inds_for_c[i]])
                  final_probs.append(probs[inds_for_c[i]])
                  final_class_inds.append(c)

        print("Final probs: ", final_probs)
        print("Final Class idx: ", final_class_inds)
        return final_boxes, final_probs, final_class_inds

    def variable_with_weight_decay(self, name, shape, wd, initializer, trainable=True):
        var = self.get_variable(name, shape=shape, dtype=tf.float32,
                    initializer=initializer, trainable=trainable)

        if wd is not None and trainable:
            # add a variable weight decay loss:
            weight_decay = wd*tf.nn.l2_loss(var)
            tf.add_to_collection("losses", weight_decay)

        return var

    def get_variable(self, name, shape, dtype, initializer, trainable=True):
        # (this wrapper function of tf.get_variable is needed because when the
        # initializer is a constant (kernel_init = tf.constant(kernel_val,
        # dtype=tf.float32)), you should not specify the shape in
        # tf.get_variable)

        if not callable(initializer):
            var = tf.get_variable(name, dtype=dtype, initializer=initializer,
                        trainable=trainable)
        else:
            var = tf.get_variable(name, shape, dtype=dtype, initializer=initializer,
                        trainable=trainable)

        return var


    def set_anchors(self):
        # NOTE! this function is taken directly from
        # github.com/BichenWuUCB/squeezeDet
        # H is the number of grid in vertical axis
        # W is the number of grid in the horizontal axis
        H = int(self.img_height / 16)   # = 45
        W = int(self.img_width / 16)    # = 80
        B = 9
        print("H,W,B: %d, %d, %d" % (H,W,B))
        anchor_shapes = np.reshape(
            [np.array(
                [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
                [ 162.,  87.], [  38.,  90.], [ 258., 173.],
                [ 224., 108.], [  78., 170.], [  72.,  43.]])]*H*W,
            (H, W, B, 2)
        )

        center_x = np.reshape(
            np.transpose(
                np.reshape(
                    np.array([np.arange(1, W+1)*float(self.img_width)/(W+1)]*H*B),
                    (B, H, W)
                ),
                (1, 2, 0)
            ),
            (H, W, B, 1)
        )

        center_y = np.reshape(
            np.transpose(
                np.reshape(
                    np.array([np.arange(1, H+1)*float(self.img_height)/(H+1)]*W*B),
                    (B, W, H)
                ),
                (2, 1, 0)
            ),
            (H, W, B, 1)
        )
        # reshaping array to N rows and 4 columns
        anchors = np.reshape(
            np.concatenate((center_x, center_y, anchor_shapes), axis=3),
            (-1, 4)
        )

        return anchors

