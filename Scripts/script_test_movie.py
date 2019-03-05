from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Scripts.root import root
import matplotlib.pyplot as plt
import numpy as np
import cv2

from Tools.MiscellaneousTools.Geometry import distance

group = 'UO'
id_exp = 43

Exps = ExperimentGroupBuilder(root).build(group)
Exps.load(['crop_limit_x', 'crop_limit_y', ])
x_lim = list(Exps.crop_limit_x.df.loc[id_exp].astype(int))
y_lim = list(Exps.crop_limit_y.df.loc[id_exp].astype(int))

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 2
detector = cv2.SimpleBlobDetector_create(params)

for id_exp in range(44, 60):
    bg_img = Exps.get_bg_img(id_exp)
    bg_img = np.sqrt(bg_img[400:600, 900:1100]).astype(np.uint8)
    # bg_img = cv2.adaptiveThreshold(bg_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.subplot(221)
    plt.imshow(bg_img, cmap='gray')
    plt.subplot(222)
    plt.hist(bg_img.ravel())
    plt.subplot(223)
    ret, bg_img = cv2.threshold(bg_img, 9.5, 255, cv2.THRESH_BINARY)
    plt.imshow(bg_img, cmap='gray')

    key_points = detector.detect(bg_img)
    ds = np.zeros((len(key_points), len(key_points)))
    for i in range(len(key_points)):
        plt.plot(key_points[i].pt[0], key_points[i].pt[1], 'o', alpha=0.5)
        for j in range(i+1, len(key_points)):
            ds[i, j] = distance(key_points[i].pt, key_points[j].pt)

    plt.subplot(224)
    plt.hist(ds[ds != 0].ravel())
    plt.show()

# mask = np.where(h[0] > 4)[0]
# d0 = h[1][mask[0]]
# d1 = h[1][mask[0]+1]
# d = np.max(ds[(ds >= d0)*(ds <= d1)])
#
# print(d)


plt.show()

