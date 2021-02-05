from __future__ import division
import numpy as np
import scipy.io
import scipy.ndimage
from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize
import cv2

def augment(img, saliencymap, rot_90=True):
    if rot_90:
        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        if angle == 270:
            img = cv2.flip(img, 0)
            saliencymap = cv2.flip(saliencymap, 0)
        elif angle == 180:
            img = cv2.flip(img, -1)
            saliencymap = cv2.flip(saliencymap, -1)
        elif angle == 90:
            img = cv2.flip(img, 1)
            saliencymap = cv2.flip(saliencymap, 1)
        elif angle == 0:
            pass
    return img, saliencymap


def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = imresize(img, (shape_r, new_cols))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = imresize(img, (new_rows,shape_c))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out

def resize_label(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = img[coord[0], coord[1]]

    return out

def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = imread(path)
        if original_image.ndim == 2:
            copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image
            original_image = copy
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims[:, :, :, ::-1]
    return ims


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = imread(path)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, :, :, 0] = padded_map.astype(np.float32)
        ims[i, :, :, 0] /= 255.0

    return ims


def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["I"]
        ims[i, :, :, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    pred = imresize(pred, (shape_r, shape_c))
    pred = pred / np.max(pred) * 255

    return pred
