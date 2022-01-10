def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj    
import torch.jit
torch.jit.script_method = script_method 
torch.jit.script = script


import tkinter as tk
from PIL import ImageTk, Image
import tkinter.font as tkFont
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2
from kl_process import *
import argparse
import numpy as np
import os
import math


class GUI:
    def __init__(self, window, args):
        self.window = window
        self.args = args
        self.ft = tkFont.Font(family='Fixdsys', size=12, weight=tkFont.BOLD)
        self.window.title('KL-Score Prediction System')
        #self.window.geometry('2600x1200')
        self.window.rowconfigure(2, minsize = 800, weight = 1)
        self.window.columnconfigure(3, minsize = 800, weight = 1)
        #create the frame structure
        # the frame for the buttons is on the top
        self.frame_b = tk.Frame(window)
        self.frame_b.pack()
        #he frame for the raw image is on the left
        self.frame_img = tk.Frame(window)
        self.frame_img.pack()
        # the frame for the raw image
        self.frame_img_raw = tk.Frame(self.frame_img)
        self.frame_img_raw.pack(side = 'left', anchor = 'n',padx=20, pady=20)
        
        # choose the first img to detect
        #self.img_path = ''
        #self.img_path = self.open_img()
    
        
        # add the button for import images
        self.b_import = tk.Button(self.frame_b, text = 'Import', font = ('Arial',12, tkFont.BOLD), width = 15, height = 1, command = self.open_img)
        #self.b_cal = tk.Button(self.frame_b, text = 'Predict', font = ('Arial',12, tkFont.BOLD), width = 15, height = 1,command = self.cal_bt)
        self.b_save = tk.Button(self.frame_b, text = 'Save', font = ('Arial',12, tkFont.BOLD), width = 15, height = 1, command = self.save_bt)
        self.b_import.pack(side = 'left')
        #self.b_cal.pack(side = 'left')
        self.b_save.pack(side = 'left')
        

        self.img_raw_label = tk.Label(self.frame_img_raw)
        self.img_raw_label.pack()
        
    
    def open_img(self):
        '''open a image to show'''
        filepath = askopenfilename(filetypes=[("Image Files", "*.*"), (" Files", "*.*")]
        )
        if not filepath:
            return
        
        self.img_path = filepath
  
        args.img_name = self.img_path
        # the main function to calculate the result
        self.raw_img, self.seg_result, self.medial_hight, self.lateral_hight, self.predict_kl = seg_class_function(self.args)
        self.backImg = draw_jpeg(self.raw_img, self.seg_result, self.medial_hight, self.lateral_hight, self.predict_kl)
        self.img_raw = ImageTk.PhotoImage(self.backImg, Image.ANTIALIAS)
        self.img_raw_label.configure(image = self.img_raw)
        self.img_raw_label.update()
            
    def get_filename(self, path, filetype):
        name = []
        for root,dirs,files in os.walk(path):
            for i in files:
                if os.path.splitext(i)[1]==filetype:
                    name.append(i)    
        return name  
    
    def cal_bt(self):
        args.img_name = self.img_path
        self.raw_img, self.seg_result, self.medial_hight, self.lateral_hight, self.predict_kl = seg_class_function(self.args)
        self.backImg = draw_jpeg(self.raw_img, self.seg_result, self.medial_hight, self.lateral_hight, self.predict_kl)
        self.img_raw = ImageTk.PhotoImage(self.backImg, Image.ANTIALIAS)
        self.img_raw_label.configure(image = self.img_raw)
        self.img_raw_label.update()
        
    
    def save_bt(self):
        img_i = os.path.split(self.img_path)[-1]
        name = asksaveasfilename(initialfile = img_i, title= 'type name of the saving result')
        file = open(name +'.txt' ,'w')
        txt = ''
        for i, s in enumerate(['R','L']):
            txt +=  'JSN, ' +s + ' knee, lat: ' + str(int(self.lateral_hight[i])) +'\n'
            txt +=  'JSN, ' +s + ' knee, med: ' + str(int(self.medial_hight[i])) +'\n'
            txt += 'KL, ' + s + ' knee: ' +str(int(self.predict_kl[i])) +'\n'
        file.write(txt)
        file.close
        self.backImg.save(name+'.png')

def main(args):
    
    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using :0.0')
        os.environ.__setitem__('DISPLAY', ':0.0')
    # set basic paramters for the window
    window = tk.Tk()
    app = GUI(window, args)
    window.mainloop()
    # combine them
    #in_final = np.append(class_out,[lateral_hight, medial_hight])
    #print(in_final)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KL classifier")
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
