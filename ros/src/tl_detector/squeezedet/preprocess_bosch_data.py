#Implementation of the preprocessing for Bosch traffic light dataset
#Base code is taken from https://github.com/fregu856/2D_detection

import cv2
import pickle
import os
import numpy as np
import random
import click
import yaml

from utilities import bbox_transform_inv


@click.command()
@click.argument('bosch_data_dir', nargs=1)
def preprocessing(bosch_data_dir, augmentation=0):

    if not os.path.isabs(bosch_data_dir):
        bosch_data_dir = os.path.join(os.getcwd(),bosch_data_dir)

    new_img_height = 375 # (the height all images fed to the model will be resized to)
    new_img_width = 1242 # (the width all images fed to the model will be resized to)
    no_of_classes = 4 # (number of object classes (cars, pedestrians, bicyclists))

    input_yaml = os.path.join(bosch_data_dir, "train.yaml")
    bosch_data = yaml.load(open(input_yaml, 'rb').read())
    train_data_dict ={}
    train_img_paths=[]
    train_img_boxes=[]
    #Reformatting bounding boxes
    for image_dict in bosch_data:

        image_path = os.path.abspath(os.path.join(os.path.dirname(input_yaml), image_dict['path']))
        bboxes=[]
        for box in image_dict['boxes']:
            w = box["x_max"] - box["x_min"]
            h = box["y_max"] - box["y_min"]
            cx = w/2
            cy = h/2
            bboxes.append([cx, cy, w,h, box['label']])
        train_data_dict[image_path]=bboxes
        train_img_paths.append(image_path)
        train_img_boxes.append(bboxes)

    mean_channel_path = os.path.join(project_dir,"data/bosch_mean_channels.pkl")
    #if not os.path.exists(mean_channel_path):
    compute_mean_channel(train_img_paths)

    # Save as dict
    save_dir = os.path.join(project_dir, "data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not augmentation:
        # Save as dict
        pickle.dump(train_data_dict, open(project_dir + "/data/bosch_dict_train_data.pkl", "wb"))
        print("train_data: ", len(train_data_dict))
    else:
        augment_train_data(train_data_dict)


def compute_mean_channel(train_img_paths):
    # compute the mean color channels of the train imgs:
    print("computing the mean color channel of the training imgs")
    no_of_train_imgs = len(train_img_paths)
    mean_channels = np.zeros((3, ))
    for step, img_path in enumerate(train_img_paths):
        if step % 100 == 0:
            print(step)

        img = cv2.imread(img_path, -1)

        img_mean_channels = np.mean(img, axis=0)
        img_mean_channels = np.mean(img_mean_channels, axis=0)

        mean_channels += img_mean_channels

    train_mean_channels = mean_channels/float(no_of_train_imgs)
    # # save to disk:
    pickle.dump(mean_channels, open(project_dir + "/data/bosch_mean_channels.pkl", "wb"))



def augment_train_data(train_data):

    # augment the train data by flipping all train imgs:
    augmented_train_img_paths = []
    augmented_train_label_paths = []
    print("Training data augmentation...")
    for step, (img_path, label_path) in enumerate(train_data):
        if step % 100 == 0:
            print(step)

        img = cv2.imread(img_path, -1)

        # flip the img and save to project_dir/data:
        img_flipped = cv2.flip(img, 1)
        img_flipped_path = img_path.split(".png")[0] + "_flipped.png"
        img_flipped_path = (data_dir + img_flipped_path.split("/image_2/")[1])
        cv2.imwrite(img_flipped_path, img_flipped)
        # save the paths to the flipped and original imgs (NOTE! the order must
        # match the order in which we append the label paths below):
        augmented_train_img_paths.append(img_flipped_path)
        augmented_train_img_paths.append(img_path)

        # modify the corresponding label file to match the flipping and save to
        # project_dir/data:
        label_flipped_path = label_path.split(".txt")[0] + "_flipped.txt"
        label_flipped_path = (data_dir + label_flipped_path.split("/label_2/")[1])
        label_flipped_file = open(label_flipped_path, "w")
        with open(label_path) as label_file:
            for line in label_file:
                splitted_line = line.split(" ")
                x_left = float(splitted_line[4])
                x_right = float(splitted_line[6])

                x_right_flipped = str(new_img_width/2 - (x_left - new_img_width/2))
                x_left_flipped = str(new_img_width/2 - (x_right - new_img_width/2))

                new_line = (splitted_line[0] + " " + splitted_line[1] + " "
                            + splitted_line[2] + " " + splitted_line[3] + " "
                            + x_left_flipped + " " + splitted_line[5] + " "
                            + x_right_flipped + " " + splitted_line[7] + " "
                            + splitted_line[8] + " " + splitted_line[9] + " "
                            + splitted_line[10] + " " + splitted_line[11] + " "
                            + splitted_line[12] + " " + splitted_line[13] + " "
                            + splitted_line[14])

                label_flipped_file.write(new_line)
        label_flipped_file.close()

        # save the paths to the flipped and original label files (NOTE! the order
        # must match the order in which we append the img paths above):
        augmented_train_label_paths.append(label_flipped_path)
        augmented_train_label_paths.append(label_path)

    # # randomly shuffle the augmented train data:
    augmented_train_data = zip(augmented_train_img_paths, augmented_train_label_paths)
    random.shuffle(augmented_train_data)
    random.shuffle(augmented_train_data)
    random.shuffle(augmented_train_data)
    random.shuffle(augmented_train_data)

    # # save the augmented train data to disk:
    train_data = augmented_train_data
    train_img_paths, train_label_paths = zip(*train_data)
    no_of_train_imgs = len(train_img_paths)
    cPickle.dump(train_label_paths,
                open(project_dir + "/data/train_label_paths.pkl", "w"))
    cPickle.dump(train_img_paths,
                open(project_dir + "/data/train_img_paths.pkl", "w"))
    # train_label_paths = cPickle.load(open(project_dir + "data/train_label_paths.pkl"))
    # train_img_paths = cPickle.load(open(project_dir + "data/train_img_paths.pkl"))
    print("number of train imgs after augmentation: %d " % len(train_data))





# read all relevant bboxes (bounding boxes) from the train labels:
# # (train_bboxes_per_img is a list of length no_of_train_imgs where each
# # element is a list containing that img's bboxes)
# train_bboxes_per_img = []
# for step, label_path in enumerate(train_label_paths):
#     if step % 100 == 0:
#         print step

#     bboxes = []
#     with open(label_path) as label_file:
#         for line in label_file:
#             splitted_line = line.split(" ")
#             bbox_class = splitted_line[0].lower().strip()
#             if bbox_class not in ["car", "cyclist", "pedestrian"]:
#                 break
#             x_min = float(splitted_line[4])
#             y_min = float(splitted_line[5])
#             x_max = float(splitted_line[6])
#             y_max = float(splitted_line[7])

#             c_x, c_y, w, h = bbox_transform_inv([x_min, y_min, x_max, y_max])
#             bboxes.append([c_x, c_y, w, h, bbox_class])

#     train_bboxes_per_img.append(bboxes)

# # # save to disk:
# cPickle.dump(train_bboxes_per_img,
#             open(project_dir + "/data/train_bboxes_per_img.pkl", "w"))

# read all relevant bboxes from the val labels:
# val_bboxes_per_img = []
# for step, label_path in enumerate(val_label_paths):
#     if step % 100 == 0:
#         print step

#     bboxes = []
#     with open(label_path) as label_file:
#         for line in label_file:
#             splitted_line = line.split(" ")
#             bbox_class = splitted_line[0].lower().strip()
#             if bbox_class not in ["car", "cyclist", "pedestrian"]:
#                 break
#             x_min = float(splitted_line[4])
#             y_min = float(splitted_line[5])
#             x_max = float(splitted_line[6])
#             y_max = float(splitted_line[7])

#             c_x, c_y, w, h = bbox_transform_inv([x_min, y_min, x_max, y_max])
#             bboxes.append([c_x, c_y, w, h, bbox_class])

#     val_bboxes_per_img.append(bboxes)

# # # save to disk for later use:
# cPickle.dump(val_bboxes_per_img,
#             open(project_dir + "/data/val_bboxes_per_img.pkl", "w"))



if __name__ == '__main__':
    global project_dir
    project_dir = os.getcwd()
    preprocessing()