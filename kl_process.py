from utils.models import *  
#from utils.datasets import *
from utils.utils import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from utils.seg_function import *
import torch
import pydicom
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from torch.nn import DataParallel
import numpy as np
import sys
import joblib 

def detect(img0):
    '''
        return the index of the detected knees
        first line is the R knee
        second line is the L knee
    '''
    weights = 'model_files/best_finalversion.pt'
    cfg = 'model_files/yolov3-tiny.cfg'
    conf_thres = 0.001
    iou_thres = 0.2
    # Initialize
    device = torch_utils.select_device(device='cpu')
    imgsz = 512
    detection_model = Darknet(cfg, imgsz)
    name = ['joint']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(name))]
    
    # load the weight
    if weights.endswith('.pt'):
        detection_model.load_state_dict(torch.load(weights, map_location=device)['model'])
    detection_model.to(device).eval()
    #Read img
    img0 = np.stack((img0,)*3,axis = -1)
    #img0 = cv2.imread(img_path) # BGR
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]] # the gain in width and height
    #padded resze
    img = letterbox(img0, new_shape = imgsz)[0]
    #covert
    img = img[:, :, :].transpose(2, 0, 1) # change BGR to RGB, 3*imgsz*imgsz
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /=255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    # to calculate the time
    pred = detection_model(img, augment=False)[0]
    #print(pred)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   multi_label=False, classes=None, agnostic=False)
    det = pred[0]
    #process detections
    if det is not None and len(det):
        # Rescale boxes from imgsz to im0 size
        s = ''
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        # Print results
        n = (det[:, -1] == 0).sum()  # detections per class
        s += '%g %ss, ' % (n, name[int(0)])  # add to string
        
        # only save two highest confidence box   
        det = reversed(det[det[:,4].argsort()])[:2,:]
        result_box=[]
        center_index = []
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            result_box = xywh
            label = '%s %.2f' % (name[int(cls)], conf)
            #plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])
            center_index.append([int(xywh[0]*img0.shape[1]), int(xywh[1]*img0.shape[0])])
        #plt.imshow(img0)
        center_index = np.array(center_index)
        center_index = center_index[center_index[:,0].argsort()]
        return center_index[:,0], center_index[:,1]

def get_input(args, side, file_type):
    ''' This function is to input image and preprocess for future use
        return: 
        seg_img: the image for segmentation
        class_img: the image for classification
        cropped_img: the image which has just been cropped
    '''


    # read image: distinguish with png or dicom type
    img_path = args.img_name
    if file_type == 'png':
        try:
            img = Image.open(img_path)
            img = np.asarray(img)
            print(img.shape)
        except:
            print("Image does not exist. (or) This type is not supported.")
            sys.exit()
    elif file_type=='dicom':
        try:
            img = read_dicom(img_path)
        except:
            print("Image does not exist. (or) This type is not supported.")
            sys.exit()
    else:
        raise ValueError(f"file fype {file_type} not supported")

    # initialize the arrays
    cropped_img = np.zeros([2,args.box_size, args.box_size])
    norm_img = np.zeros([2,args.box_size, args.box_size]).astype(np.float32)
    seg_img = np.zeros([2,1, args.box_size, args.box_size]).astype(np.float32)
    class_img = np.zeros([2,args.box_size, args.box_size]).astype(np.float32)
    #detect the joint and crop the img
    for i,s in enumerate(side):
        joint_x, joint_y = detect(img)
        cropped_img[i] = crop(joint_y[i], joint_x[i], img, args.box_size)
        
        # prepare img for classification
        class_img[i] = normalize(cropped_img[i])

        
        # prepare img for segmentation
        norm_img[i] = normalize(cropped_img[i])
        norm_img[i] = shapen_Laplacian(norm_img[i])
        seg_img[i] = np.expand_dims(norm_img[i], axis=0)
        seg_img[i] = normalize_seg(seg_img[i])  
    return seg_img, class_img, cropped_img, img



def seg_cal_narrow(args, x, cropped_img, s):
    ''' This function is to calculate Jiont space naroowing distance
        x: mask after segmentation
        cropped_img: image after crop
        s: side
    '''
    device = torch_utils.select_device(device='cpu')
    unet = UNet(in_channels=1, out_channels=args.mode, init_features=16)
    unet.to(device)
    unet = DataParallel(unet)
    unet.load_state_dict(torch.load("model_files/unet_new.pt", map_location=device))
    unet.eval()
    with torch.set_grad_enabled(False):
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float)
        y = unet(x)
        y = y.cpu().numpy() 
        x = gray_to_rgb(np.squeeze(cropped_img))  
        y = np.squeeze(y)
        for i in range(y.shape[0]):
            y[i] = fill_holes(y[i])
            x = outline_mask(x, y[i], mask_color=[255,0 , 0])
        lateral_hight, medial_hight, lateral_index, medial_index = manual_height(y, narrow_type = args.narrow_type, side=s, file_name=None)
        # draw the narrow lines
        x = draw_height(x,lateral_index[0],lateral_index[1],medial_index[0],medial_index[1],lateral_index[2] , medial_index[2], mask_color = [0,0,255])
    return lateral_hight, medial_hight, x

def classify_single_view(img):
    '''get initial classification result by kl classification model
    '''
    TRANSFORMS = transforms.Compose([
        transforms.ToPILImage(mode=None),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
    device = torch_utils.select_device(device='cpu')
    model = SingleVgg16(channels=1, classes=5, filters=32)
    model.to()
    model = DataParallel(model)
    model.load_state_dict(torch.load("model_files/detect.pt", map_location=device))
    model.eval()
    #print(img.shape)
    img= torch.tensor(img)
    img = TRANSFORMS(img)
    #img = torch.tensor(img, dtype=torch.float)
    data = img.view(1, 1, 256, 256)
    outputs = model(data) 
    _, predicted = torch.max(outputs.data, 1)
    outputs = np.squeeze(outputs.cpu().detach().numpy())
    predict_kl = predicted.cpu().numpy()[0]
    print('predict_kl',predict_kl)
    return outputs, predict_kl


def seg_class_function(args):
    ''' combine the classifciation and segmentation results
    '''
    side = ['R', 'L']
    if os.path.splitext(args.img_name)[-1]=='.png':
        file_type = 'png'
    else:
        file_type = 'dicom'

    # read image and preprocess
    seg_img, class_img, cropped_img, raw_img = get_input(args, side, file_type)
    #print(class_img.shape)
    lateral_hight = np.zeros(2)
    medial_hight = np.zeros(2)
    seg_result = np.zeros((*cropped_img.shape,3)).astype(np.uint8)
    predict_kl = np.zeros(2)


    #load random forest model
    clf = joblib.load('model_files/random_forest.pkl')

    # get output from classification
    for i,s in enumerate(side):
        class_out, predict_kl[i] = classify_single_view(class_img[i])
        # get output from segmentation
        print(class_out)
        lateral_hight[i], medial_hight[i], seg_result[i] = seg_cal_narrow(args,seg_img[i],cropped_img[i],s)
        lat_input = lateral_hight[i]
        med_input = medial_hight[i]
        pre_x = np.append(class_out,[med_input,lat_input]).reshape(1, -1)
        # random forest classification
        pre_kl = clf.predict(pre_x)
        lateral_hight[i]= dis_to_grade(lateral_hight[i])
        medial_hight[i] = dis_to_grade(medial_hight[i])
        predict_kl[i] = pre_kl
    return raw_img, seg_result, medial_hight, lateral_hight, predict_kl


def draw_jpeg(raw_img, seg_result, medial_hight, lateral_hight, predict_kl):
    backImg = Image.new('RGB', (1800,800), 'white')

    # paste the raw image into jpeg
    raw_img = np.stack((raw_img,)*3,axis = -1).astype(np.uint8)
    raw_img_show = Image.fromarray(raw_img, 'RGB')
    backImg.paste(raw_img_show.resize((522,644)),(60,100))
    draw = ImageDraw.Draw(backImg, 'RGB')
    ft = ImageFont.truetype("utils/FontsFree-Net-arial-bold.ttf",28)
    draw.text((220, 40), "Raw Dicom Image", (0,0,0), font=ft)

    # paste the seg result into jpeg
    seg_result_show_R = Image.fromarray(seg_result[0], 'RGB')
    backImg.paste(seg_result_show_R.resize((435,435)),(700,100))
    draw = ImageDraw.Draw(backImg, 'RGB')
    ft = ImageFont.truetype("utils/FontsFree-Net-arial-bold.ttf",28)
    draw.text((840, 40), "Right Knee", (0,0,0), font=ft)

    seg_result_show_L = Image.fromarray(seg_result[1], 'RGB')
    backImg.paste(seg_result_show_L.resize((435,435)),(1200,100))
    draw = ImageDraw.Draw(backImg, 'RGB')
    draw.text((1360, 40), "Left Knee", (0,0,0), font=ft)

    # write the predict result
    JSN_show_R = 'JSN (LAT): ' + str(int(lateral_hight[0])) + ' ' + 'JSN(MED): '+ str(int(medial_hight[0]))
    JSN_show_L = 'JSN (MED): ' + str(int(medial_hight[1])) + ' ' + 'JSN(LAT): '+ str(int(lateral_hight[1]))
    predict_kl_show_R = 'Predicted KL score (R): ' + str(int(predict_kl[0]))
    predict_kl_show_L = 'Predicted KL score (L): ' + str(int(predict_kl[1]))
    draw.text((700, 600), JSN_show_R, (0,0,0), font=ft)
    draw.text((1200, 600), JSN_show_L, (0,0,0), font=ft)
    draw.text((700, 700), predict_kl_show_R, (0,0,0), font=ft)
    draw.text((1200, 700), predict_kl_show_L, (0,0,0), font=ft)
    return backImg