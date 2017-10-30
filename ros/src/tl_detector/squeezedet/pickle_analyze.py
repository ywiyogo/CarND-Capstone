import os,pickle
import cv2
import click
from utilities import vis_gt_bboxes, draw_bboxes, resize_bbox_fcenter

@click.command()
@click.argument('dataset_dir', nargs=1)
@click.argument('keyword', nargs=1)
def analyze(dataset_dir, keyword):
    pickle_dir = os.path.join(dataset_dir,"pickles")
    data_dict = pickle.load(open(os.path.join(pickle_dir,"bosch_dict_train_data.pkl"), "rb"))

    with open(os.path.join(pickle_dir,"bosch_mean_channels.pkl"), "rb") as f:
        train_mean_channels= pickle.load(f, encoding='bytes')

    distrib = {"Red":0, "Yellow":0, "Green":0, "off":0}
    print("Train mean channel: ",train_mean_channels)
    img_height = 720
    img_width = 1280

    for step, (path, bboxes) in enumerate(data_dict.items()):
        if keyword in path:
            abs_path = os.path.join(dataset_dir, path)
            img_orig = cv2.imread(abs_path)
            img = cv2.resize(img_orig, (img_width, img_height))

            print("Original image: ",img_orig.shape)
            print("After resize: ", img.shape)
            hscale = img.shape[0] / img_orig.shape[0]
            wscale = img.shape[1] / img_orig.shape[1]

            print(hscale, "" , wscale)
            classes= []
            newbox=[]
            for box in bboxes:
                #cx, cx, w, h
                nbox = [box[0],box[1], box[2], box[3]]
                newbox.append(resize_bbox_fcenter(img_orig.shape, img.shape, nbox))
                if "Red" in box[4]:
                    distrib["Red"] = distrib["Red"]+1
                    classes.append(0)
                elif "Yellow" in box[4]:
                    distrib["Yellow"] = distrib["Yellow"]+1
                    classes.append(1)
                elif "Green" in box[4]:
                     distrib["Green"] = distrib["Green"]+1
                     classes.append(2)
                else:
                    distrib["off"] = distrib["off"]+1
                    classes.append(3)


            #bbox_img = vis_gt_bboxes(abs_path, bboxes)
            bbox_img = draw_bboxes(img, newbox, classes)
            assert not bbox_img is None
            if 1:
                print("BBox: ", bboxes)
                cv2.imshow(path, bbox_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    print(distrib)
    #{'Yellow': 588, 'Green': 6480, 'Red': 4951}

if __name__ == '__main__':
    analyze()