def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj    
import torch.jit
torch.jit.script_method = script_method 
torch.jit.script = script

import cv2
from kl_process import *
import argparse
import numpy as np
import os
import math




def main(args):
    raw_img, seg_result, medial_hight, lateral_hight, predict_kl = seg_class_function(args)
    if not args.png_path=='None':
        # draw and save the result jpeg
        #Image.fromarray(seg_result[i], mode = 'RGB').save(args.jpg_path+'/seg_'+s+'.png')
        backImg = draw_jpeg(raw_img, seg_result, medial_hight, lateral_hight, predict_kl)
        backImg.save(args.png_path)
    if not args.txt_path=='None':    
        result_txt = ''
        for i,s in enumerate(['R','L']):     
            result_txt = result_txt + 'JSN, %s knee, lat: %d\n'%(s,int(lateral_hight[i]))
            result_txt = result_txt + 'JSN, %s knee, med: %d\n'%(s,int(medial_hight[i]))
            result_txt = result_txt + 'KL, %s knee: %d\n'%(s,int(predict_kl[i]))
        file = open(args.txt_path,'w')
        file.write(result_txt)
        file.close()
        # combine them
        #in_final = np.append(class_out,[lateral_hight, medial_hight])
    #print(in_final)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KL classifier")
    parser.add_argument("--img-name", default="None", type=str, help="Path to test image")
    parser.add_argument("--png-path", default="None", type=str, help="Path to save image")
    parser.add_argument("--txt-path", default="None", type=str, help="Path to save image")
    parser.add_argument("--narrow-type", default="lower_upper_mean", type=str, help = "the method to calculate narrowing distance")

    parser.add_argument(
        "--box-size",
        type=int,
        default=672,
        help="box size",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default='./',
        help="save path",
    )
    parser.add_argument(
        "--shape",
        type=tuple,
        default=(672, 672), 
        help="image size",
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        help="bone mode",
    )
    args = parser.parse_args()
    #print(f"Running in mode: {mode}")
    main(args)
