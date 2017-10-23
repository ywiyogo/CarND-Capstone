import os,pickle
import cv2
import click
from utilities import vis_gt_bboxes

@click.command()
@click.argument('dataset_dir', nargs=1)
def analyze(dataset_dir, augmentation=0):
    pickle_dir = os.path.join(dataset_dir,"pickles")
    data_dict = pickle.load(open(os.path.join(pickle_dir,"bosch_dict_train_data.pkl"), "rb"))

    for step, (path, bboxes) in enumerate(data_dict.items()):
        abs_path = os.path.join(dataset_dir, path)
        print(bboxes)
        classes= []
        bbox_img = vis_gt_bboxes(abs_path, bboxes)
        assert not bbox_img is None
        print("BBox: ", bboxes) 
        cv2.imshow("Test Image", bbox_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    analyze()