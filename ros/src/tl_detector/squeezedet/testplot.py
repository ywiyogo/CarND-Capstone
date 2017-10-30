import matplotlib.pyplot as plt
import numpy as np
import cv2
import random as rd
H = int(720 / 16)   # = 45
W = int(1280 / 16)    # = 80
B = 9
print("H,W,B: %d, %d, %d" % (H,W,B))
anchor_shapes = np.reshape(
    [np.array(
        [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
        [ 162.,  87.], [  38.,  90.], [ 258., 173.],
        [ 224., 108.], [  78., 170.], [  72.,  43.]])]*H*W,
    (H, W, B, 2)
)
fig, (ax1,ax2) = plt.subplots(2)
a = np.array(
        [[ 20., 32.], [  36.,  36.],  [  45.,  90.],
         [ 72., 43.], [ 78., 140.], [ 120., 200.],
         [ 115.,  59.], [ 162.,  87.], [ 258., 173.]])
img = cv2.imread("/media/yongkie/YSDC/dataset_train_rgb_y2/rgb/train/2015-10-05-10-55-33_bag/30252.png", -1)
for b in a:

    x_left = int(200 - b[0]/2)
    y_bottom = int(200 + b[1]/2)
    x_right = int(200 + b[0]/2)
    y_top = int(200 - b[1]/2)
    print(x_left, x_right, y_bottom, y_top)
    color = (int(rd.random()*255), int(rd.random()*255), int(rd.random()*255))
    cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom),
                        color, 2)
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(a)
y,x = zip(*a)
ax1.scatter(x,y)

center_x = np.reshape(
    np.transpose(
        np.reshape(
            np.array([np.arange(1, W+1)*float(1280)/(W+1)]*H*B),
            (B, H, W)
        ),
        (1, 2, 0)
    ),
    (H, W, B, 1)
)
center_y = np.reshape(
    np.transpose(
        np.reshape(
            np.array([np.arange(1, H+1)*float(720)/(H+1)]*W*B),
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
x=[]
y=[]
for idx, anch in enumerate(anchors):
    if idx > len(anchors)/2:
        break
    x.append(anch[0])
    y.append(anch[1])
ax2.scatter(x, y)
plt.show()