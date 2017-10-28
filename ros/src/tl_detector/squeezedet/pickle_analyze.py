import os,pickle
import cv2
import click
from utilities import vis_gt_bboxes

@click.command()
@click.argument('dataset_dir', nargs=1)
def analyze(dataset_dir, augmentation=0):
    pickle_dir = os.path.join(dataset_dir,"pickles")
    data_dict = pickle.load(open(os.path.join(pickle_dir,"bosch_dict_train_data.pkl"), "rb"))

    distrib = {"Red":0, "Yellow":0, "Green":0}

    for step, (path, bboxes) in enumerate(data_dict.items()):
        if "left001" in path:
            abs_path = os.path.join(dataset_dir, path)
            print(bboxes)
            classes= []
            for box in bboxes:
                if "Red" in box[4]:
                    distrib["Red"] = distrib["Red"]+1
                elif "Yellow" in box[4]:
                    distrib["Yellow"] = distrib["Yellow"]+1
                elif "Green" in box[4]:
                     distrib["Green"] = distrib["Green"]+1

            bbox_img = vis_gt_bboxes(abs_path, bboxes)
            assert not bbox_img is None
            if 1:
                print("BBox: ", bboxes)
                cv2.imshow(path, bbox_img)
                cv2.waitKey(0)

    print(distrib)
    #{'Yellow': 588, 'Green': 6480, 'Red': 4951}

if __name__ == '__main__':
    analyze()