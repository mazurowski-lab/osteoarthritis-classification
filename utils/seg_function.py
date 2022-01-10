import argparse
import os
import pandas as pd
import numpy as np
import torch
from skimage.transform import resize
from skimage import morphology
from skimage.filters import roberts
from utils.models import UNet
from scipy import ndimage as ndi
from torch.nn import DataParallel
from PIL import ImageFile, Image
import matplotlib.pyplot as plt
import cv2
import utils
import argparse
from scipy.signal import find_peaks


def get_ROI(masks, file_name=None):
    """
     get region of interest
     input:
         mask of bones -> np.ndarray (c, h, w)
     output:
        x, y, w, h corrds of box with ROI (top left corner with w and h)
    """

    if len(masks.shape) == 3:
        mask = masks[0].copy()
        for c in range(1, masks.shape[0]):
            mask += masks[c]
    elif len(masks.shape) == 2:
        mask = masks
    else:
        raise UserWarning("Unsupported shape of mask")

    # 1. We are taking y_peak with max value from upper and lower bone
    # -> (y_top, y_bottom)
    # 2. Base on y_top and y_bottom we take most left and right non-zero cords ->
    #  -> (x_top_left, x_top_right, x_bottom_left, x_bottom_right)
    # 3. We are taking most inner (center one) from
    # (x_top_left and x_bottom_left) and (x_top_right and x_bottom_right)
    # -> x_left, x_right
    # our corrds are (
    # y == y_top, h == abs(y_top - y_bottom), x = x_left, w = x_right - x_left
    try:
        y_upper = np.sum(masks[0], axis=1)
        ys_top, _ = find_peaks(y_upper)

        y_lower = np.sum(masks[1], axis=1)
        ys_bottom, _ = find_peaks(y_lower)

        y_top = max(ys_top, key=lambda y: y_upper[y])
        y_bottom = max(ys_bottom, key=lambda y: y_lower[y])

        x_top_left, x_top_right = -1, -1
        for idx, x in enumerate(mask[y_top, :]):
            if x > 0.5:
                if x_top_left == -1:
                    x_top_left = idx
                else:
                    x_top_right = idx

        x_bottom_left, x_bottom_right = -1, -1
        for idx, x in enumerate(mask[y_bottom, :]):
            if x > 0.5:
                if x_bottom_left == -1:
                    x_bottom_left = idx
                else:
                    x_bottom_right = idx

        x_left = -1
        if x_top_left > x_bottom_left:
            x_left = x_top_left
        else:
            x_left = x_bottom_left

        x_right = -1
        if x_top_right < x_bottom_right:
            x_right = x_top_right
        else:
            x_right = x_bottom_right

        y = y_top
        h = y_bottom - y_top
        x = x_left
        w = x_right - x_left
    except:
        return 0,0,672,672
    return x, y, w, h


def first_non_zero(iterable, treshold=0.5):
    for idx, value in enumerate(iterable):
        if value > treshold:
            return idx
    return -1


def last_non_zero(iterable, treshold=0.5):
    last_idx = -1
    for idx, value in enumerate(iterable):
        if value > treshold:
            last_idx = idx
    return last_idx
    

def gamma_img(obj,gamma = 2):
    fi = obj/ 255
    out = np.power(fi, gamma)
    return out*255

    
def normalize_seg(x, shape = (672,672)):
    mean_raw = np.mean(x[0,336-50:336,336-50:336])
    #print(mean_raw)
    if mean_raw>250:
      alpha = 5
    elif mean_raw>245:
      alpha = 2.5
    elif mean_raw<200:
      alpha = 1
    else:
      alpha = 2
    x = gamma_img(x,alpha)/255
    #x = resize(x, (1, shape[0],shape[1]))
    return x

def draw_height(x, left_lower_y, left_upper_y, right_lower_y, right_upper_y,left_pick, right_pick, mask_color):
    x[max(0,left_upper_y):left_lower_y,left_pick] = mask_color
    x[max(0,right_upper_y):right_lower_y,right_pick] = mask_color
    return x

def fill_holes(mask): 
    ''' 
    this function is to fill the smalls holes after segmentation
    '''
    mask = np.round(mask)
    mask = ndi.binary_fill_holes((mask).astype(int))
    cleaned = morphology.remove_small_objects(mask, min_size = 25000, connectivity = 1)
    return cleaned
    
def manual_height(masks, narrow_type, side, file_name=None):
    '''
    This function is used to calculate the distance between joints
    '''
    # mask_copy = masks.copy()
    masks[0] = fill_holes(masks[0])
    masks[1] = fill_holes(masks[1])
    upper_bone, lower_bone = masks[0], masks[1]
    bbox = get_ROI(masks, file_name=None)
    x, y, w, h = bbox
    upper_bone[0:y, :] = 0.0
    upper_bone[:,0:x] = 0.0
    upper_bone[:, x + w:] = 0.0
    lower_bone[y+h+20:,:] = 0.0
    lower_bone[:,0:x] = 0.0
    lower_bone[:, x + w:] = 0.0
    mid = int(x + w/2)
    left_max = 0
    left_min = 672
    # find the left part
    l_h_array = []
    l_lower_y_array = []
    l_upper_y_array = []
    l_x_array = []
    for x_index in range(x+10,mid-80):
      left_lower_y = first_non_zero(lower_bone[:, x_index])
      left_upper_y = last_non_zero(upper_bone[:, x_index])
      l_h_array.append(max(left_lower_y - left_upper_y,0))
      l_lower_y_array.append(left_lower_y)
      l_upper_y_array.append(left_upper_y)
      l_x_array.append(x_index)
    l_sort_index = np.array(l_h_array).argsort()
    # min_index is sort_index[0], max_index is sort_index[-1], median_index is sort_index[int(len(l_h_array)/2)]
    
    # find the right part
    right_max = 0
    right_min = 672
    r_h_array = []
    r_lower_y_array = []
    r_upper_y_array = []
    r_x_array = []    
    for x_index in range(mid+80,x+w-10):
      right_lower_y = first_non_zero(lower_bone[:, x_index])
      right_upper_y = last_non_zero(upper_bone[:, x_index])
      r_lower_y_array.append(right_lower_y)
      r_upper_y_array.append(right_upper_y)
      r_h_array.append(max(right_lower_y - right_upper_y,0))
      r_x_array.append(x_index)
    r_sort_index = np.array(r_h_array).argsort()
    
    # choose the x_index depends on the narrow type
    if narrow_type == 'max':
        l_choose_index = l_sort_index[-1]
        r_choose_index = r_sort_index[-1]
    elif narrow_type == 'min':
        l_choose_index = l_sort_index[0]
        r_choose_index = r_sort_index[0]
    elif narrow_type == 'median':
        l_choose_index = l_sort_index[int(len(l_h_array)/2)]
        r_choose_index = r_sort_index[int(len(r_h_array)/2)]
    elif narrow_type == 'mean':
        l_mean = sum(l_h_array)/len(l_h_array)
        r_mean = sum(r_h_array)/len(r_h_array)
        l_h_array = np.array(l_h_array)
        r_h_array = np.array(r_h_array)
        l_choose_index = (np.abs(l_h_array-l_mean)).argmin()
        r_choose_index = (np.abs(r_h_array-r_mean)).argmin()
    elif narrow_type == 'min_max_mean':
        l_mean = (l_sort_index[0]+l_sort_index[-1])/2
        r_mean = (r_sort_index[0]+r_sort_index[-1])/2
        l_h_array = np.array(l_h_array)
        r_h_array = np.array(r_h_array)
        l_choose_index = (np.abs(l_h_array-l_mean)).argmin()
        r_choose_index = (np.abs(r_h_array-r_mean)).argmin()
    elif narrow_type =='lower_upper_bone':
        histogram = np.sum(upper_bone, axis=0)
        l_choose_index = histogram[x+10:mid-80].argmax()
        r_choose_index = histogram[mid+80:x+w-10].argmax()
    elif narrow_type =='lower_upper_mean':
        histogram = np.sum(upper_bone, axis=0)
        l_low_index = histogram[x+10:mid-80].argmax()
        r_low_index = histogram[mid+80:x+w-10].argmax()
        l_mean = sum(l_h_array[max(l_low_index-20,0):min(l_low_index+20,len(l_h_array))])/len(l_h_array[max(l_low_index-20,0):min(l_low_index+20,len(l_h_array))])
        r_mean = sum(r_h_array[max(r_low_index-20,0):min(r_low_index+20,len(r_h_array))])/len(r_h_array[max(r_low_index-20,0):min(r_low_index+20,len(r_h_array))])
        l_h_array = np.array(l_h_array)
        r_h_array = np.array(r_h_array)
        l_choose_index = (np.abs(l_h_array-l_mean)).argmin()
        r_choose_index = (np.abs(r_h_array-r_mean)).argmin()
    elif narrow_type =='lower_upper_max':
        histogram = np.sum(upper_bone, axis=0)
        l_low_index = histogram[x+10:mid-80].argmax()
        r_low_index = histogram[mid+80:x+w-10].argmax()
        l_choose_index = np.array(l_h_array[max(l_low_index-20,0):min(l_low_index+20,len(l_h_array))]).argmax() + max(l_low_index-20,0)
        r_choose_index = np.array(r_h_array[max(r_low_index-20,0):min(r_low_index+20,len(r_h_array))]).argmax() + max(r_low_index-20,0)
    else: 
        l_choose_index = 100
        r_choose_index = 400
    h_left = l_h_array[l_choose_index]
    index_left = np.array([l_lower_y_array[l_choose_index],l_upper_y_array[l_choose_index],l_x_array[l_choose_index]])
    h_right = r_h_array[r_choose_index]
    index_right = np.array([r_lower_y_array[r_choose_index],r_upper_y_array[r_choose_index],r_x_array[r_choose_index]])
    if side == "R":
        lateral_hight = h_left
        lateral_index = index_left      
        medial_hight = h_right
        medial_index = index_right
    elif side == "L":
        lateral_hight = h_right
        lateral_index = index_right
        medial_hight = h_left
        medial_index = index_left
    else:
        raise UserWarning("Invalid side argument -> {}".format(side))
    print(lateral_hight,medial_hight)
    return  lateral_hight, medial_hight, lateral_index, medial_index
    
    
def draw_height(x, left_lower_y, left_upper_y, right_lower_y, right_upper_y,left_pick, right_pick, mask_color):
    x[max(0,left_upper_y):left_lower_y,left_pick-1: left_pick+1] = mask_color
    x[max(0,right_upper_y):right_lower_y,right_pick-1:right_pick+1] = mask_color
    return x


def dis_to_grade(dis):
    '''
    this functiion is to transfer the joint space narrowing distance to 0-3 grade
    '''
    #print(dis)
    if dis<=7:
        grade = 3
    elif dis>7 and dis<=16:
        grade = 2
    elif dis>16 and dis<=22:
        grade = 1
    else:
        grade = 0
    return grade


def gray_to_rgb(image):
    """
    change the gray image to rgb image
    """
    width, height = image.shape
    image = normalize(image)
    rgb_image = np.empty((width, height, 3), dtype=np.uint8)
    rgb_image[:, :, 2] = rgb_image[:, :, 1] = rgb_image[:, :, 0] = image
    return rgb_image

def normalize(obj):
    assert isinstance(obj, np.ndarray)
    max = np.percentile(obj, 98)
    min = np.percentile(obj, 2)
    new = obj - min
    new /= (max - min)
    new = np.where(new > 0, new, 0)
    new = np.where(new < 1, new, 1)
    result = new * 255
    result = np.round(result).astype((int))
    return result.astype(np.float32)


def outline_mask(image, mask, mask_color):
    """
    get the image with the mask(default color = green)
    """
    mask = np.round(mask)
    pixel_cor_y, pixel_cor_x = np.nonzero(mask)
    for each_y, each_x in zip(pixel_cor_y, pixel_cor_x):
        if 0.0 < np.mean(mask[max(0, each_y - 1): each_y + 2, max(0, each_x - 1):each_x + 2]) < 1.0:
            image[max(0, each_y-1): each_y + 1, max(0, each_x-1): each_x + 1] = mask_color
    return image

def shapen_Laplacian(in_img):
    I = in_img.copy()
    L = cv2.Laplacian(I, -1)
    a = 0.3
    O = cv2.addWeighted(I, 1, L, a, 0)
    O[O > 255] = 255
    O[O < 0] = 0
    return O