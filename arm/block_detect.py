import cv2
import numpy as np
import matplotlib.path as mplPath
from copy import deepcopy

im = cv2.imread('depth_img_cropped.png')
im = im[..., 0]
cv2.imshow('orig', im)
orig_min = im.min()
orig_max = im.max()
orig_mean = im.mean()
print(im.shape)
print(orig_min, orig_max, orig_mean)

im = cv2.fastNlMeansDenoising(im, templateWindowSize=3, searchWindowSize=21, h=0)
kernel = np.ones((5, 5), dtype=np.uint8)
kernel[1, 1] = 0
ret, thresh = cv2.threshold(im, 75, 255, 0)
ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours = [i for i in contours if i.shape[0] > 50 and i.shape[0] < 600]

cv2.imshow('contours', ret_image)
cv2.waitKey(0)

print('Number of contours', len(contours))
boxes = []
contours_draw = []
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, .04 * peri, True)
    if len(approx) >= 4 and len(approx) <= 6:
        contours_draw.append(contour)
        minRect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(minRect)
        box = np.intp(box)
        area = .5*np.abs(np.dot(box[:, 0], np.roll(box[:, 1], 1)) - np.dot(box[:, 1], np.roll(box[:, 0], 1)))
        boxes.append((area, box))
boxes = sorted(boxes, reverse=True)
filtered_boxes = []
for i, (area, box_1) in enumerate(boxes):
    okay = True
    box_1_p = mplPath.Path(box_1, closed=False)
    for area, box_2 in boxes[i+1: ]:
        box_2_p = mplPath.Path(box_2, closed=False)
        if box_1_p.contains_path(box_2_p):
            okay = False
            break
    if okay:
        filtered_boxes.append(box_1)
print('Number of boxes, before and after:', len(boxes), len(filtered_boxes))

new_contours_draw = []
new_boxes = []
new_box_sizes = []
for filtered_box in filtered_boxes:
    new_img = np.zeros_like(im)
    cv2.drawContours(new_img, [filtered_box], 0,1, thickness=-1)
    im_2_filtered_box = im * new_img
    ret, thresh = cv2.threshold(im_2_filtered_box, im_2_filtered_box.max() - 5, 255, 0)
    ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [i for i in contours if i.shape[0] > 50 and i.shape[0] < 600]
    contour = contours[0]
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, .04 * peri, True)
    if len(approx) >= 4 and len(approx) <= 6:
        new_contours_draw.append(contour)
        minRect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(minRect)
        box = np.intp(box)
        area = .5*np.abs(np.dot(box[:, 0], np.roll(box[:, 1], 1)) - np.dot(box[:, 1], np.roll(box[:, 0], 1)))
        new_boxes.append(box)


for filtered_box in new_boxes:
    cv2.drawContours(im, [filtered_box], 0, 3)
cv2.imshow('boxes', im)
cv2.waitKey(0)
