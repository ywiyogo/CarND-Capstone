#Implementation of the preprocessing for Bosch traffic light dataset
#Base code is taken from https://github.com/fregu856/2D_detection

import cv2
import pickle
import os
import numpy as np
import random
import click
import yaml

from utilities import bbox_transform_inv, vis_gt_bboxes

DEBUG=0
img_height = 720 # (the height all images fed to the model will be resized to)
img_width = 1280 # (the width all images fed to the model will be resized to)

@click.command()
@click.argument('bosch_data_dir', nargs=1)
def preprocessing(bosch_data_dir, augmentation=1):
    if not os.path.isabs(bosch_data_dir):
        bosch_data_dir = os.path.join(os.getcwd(),bosch_data_dir)


    no_of_classes = 4 # (number of object classes (cars, pedestrians, bicyclists))

    input_yaml = os.path.join(bosch_data_dir, "train.yaml")
    bosch_data = yaml.load(open(input_yaml, 'rb').read())
    train_data_dict ={}
    train_img_paths=[]
    train_img_boxes=[]
    #Reformatting bounding boxes
    for image_dict in bosch_data:

        image_dict['path']
        bboxes=[]
        for box in image_dict['boxes']:
            # YW NOTE: ignore occluded box due to minus value in anchor calculation
            if not box['occluded']:
                w = box["x_max"] - box["x_min"]
                h = box["y_max"] - box["y_min"]
                cx = box["x_min"] + w/2
                cy = box["y_min"] + h/2
                bboxes.append([cx, cy, w,h, box['label']])
            else:
                print("Note: ignored occluded box in ", image_dict['path'])
        train_data_dict[image_dict['path']]=bboxes
        train_img_paths.append(image_dict['path'])
        train_img_boxes.append(bboxes)
        
        if DEBUG:
            abs_path = os.path.join(bosch_data_dir, image_dict['path'])
            bbox_img = vis_gt_bboxes(abs_path, bboxes)
            assert not bbox_img is None
            print("BBox: ", bboxes) 
            cv2.imshow("pics", bbox_img)
            cv2.waitKey(0)

    # Save as dict
    save_dir = os.path.join(bosch_data_dir, "pickles")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mean_channel_path = os.path.join(save_dir,"bosch_mean_channels.pkl")
    

    if not augmentation:
        # Save as dict
        pickle.dump(train_data_dict, open(os.path.join(save_dir, "bosch_dict_train_data.pkl"), "wb"))
        
    else:
        augment_train_data(bosch_data_dir, bosch_data, train_data_dict)
    #compute_mean_channel(bosch_data_dir, train_img_paths)
    print("Original number of images: ", len(train_img_paths))
    

def compute_mean_channel(data_dir, train_img_paths):
    # compute the mean color channels of the train imgs:
    print("computing the mean color channel of %d training imgs" % len(train_img_paths))
    no_of_train_imgs = len(train_img_paths)
    mean_channels = np.zeros((3, ))
    for step, img_path in enumerate(train_img_paths):
        if step % 500 == 0:
            print(step)

        img = cv2.imread(os.path.join(data_dir, img_path), -1)
        assert not img is None
        img_mean_channels = np.mean(img, axis=0)
        img_mean_channels = np.mean(img_mean_channels, axis=0)

        mean_channels += img_mean_channels

    train_mean_channels = mean_channels/float(no_of_train_imgs)
    # # save to disk:
    pickle.dump(mean_channels, open(os.path.join(data_dir, "pickles/bosch_mean_channels.pkl"), "wb"))



def augment_train_data(bosch_data_dir, bosch_data, train_data_dict):

    # augment the train data by flipping all train imgs:
    # Save as dict
    save_dir = os.path.join(bosch_data_dir, "augmented")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    augmented_train_img_paths = []
    augmented_train_bboxes = []
    print("Training data augmentation...")
    for step, image_dict in enumerate(bosch_data):
        if step % 5 == 0:
            img_path = image_dict['path']
            # skipped image with no bbox
            if len(image_dict['boxes']) <1:
                continue
            # Note: img_path is a relative path!
            img = cv2.imread(os.path.join(bosch_data_dir,img_path), -1)
            assert not img is None

            # flip the img and save to project_dir/data:
            img_flipped = cv2.flip(img, 1)
            img_flipped_path =  img_path.split(".png")[0] + "_flipped.png"
            img_flipped_path = os.path.join(save_dir, os.path.basename(img_flipped_path))
            cv2.imwrite(img_flipped_path, img_flipped)
            if step % 100 == 0:
                print("Flip path: ",img_flipped_path)
            # save the paths to the flipped and original imgs (NOTE! the order must
            # match the order in which we append the label paths below):
            augmented_train_img_paths.append(img_flipped_path)
            augmented_train_img_paths.append(img_path)

            # modify the corresponding label file to match the flipping and save to
            # project_dir/data:
            augmented_bboxes=[]
            for box in image_dict['boxes']:
                if not box['occluded']:
                    x_min = box["x_min"]
                    x_max = box["x_max"]

                    x_max_flipped = img_width/2 - (x_min -img_width/2)
                    x_min_flipped = img_width/2 - (x_max -img_width/2)
                    w = x_max_flipped - x_min_flipped
                    h = box["y_max"] - box["y_min"]
                    cx = x_min_flipped + w/2
                    cy = box["y_min"] + h/2
                    augmented_bboxes.append([cx, cy, w,h, box['label']])

            # save the paths to the flipped and original label files (NOTE! the order
            # must match the order in which we append the img paths above):
            train_data_dict[img_flipped_path]=augmented_bboxes
            if DEBUG:
                abs_path = os.path.join(bosch_data_dir, img_flipped_path)
                bbox_img = vis_gt_bboxes(abs_path, augmented_bboxes)
                assert not bbox_img is None 
                cv2.imshow("Augmented", bbox_img)
                cv2.waitKey(0)

    pickle.dump(train_data_dict,
                open(os.path.join(bosch_data_dir, "pickles/bosch_dict_train_data.pkl"), "wb"))

    # train_label_paths = cPickle.load(open(project_dir + "data/train_label_paths.pkl"))
    # train_img_paths = cPickle.load(open(project_dir + "data/train_img_paths.pkl"))
    print("number of train imgs after augmentation: %d " % len(train_data_dict))

if __name__ == '__main__':
    global project_dir, bosch_data_dir
    project_dir = os.getcwd()
    preprocessing()
