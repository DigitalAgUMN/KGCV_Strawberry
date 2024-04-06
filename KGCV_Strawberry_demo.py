# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:25:03 2024

@author: yang8460

Demo of the KGCV-Strawberry framework, figure 17 in the manuscript
"""
import os
import KGCV_util as util
import datetime
import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import pandas as pd
import json

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
def cal_IoU(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects (left, top, right, bottom)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0

def showImgBbox(img,bboxes):
    fig = plt.figure()
    plt.imshow(img.cpu().numpy().transpose(1,2,0))
    currentAxis=plt.gca()
   
    for bbox in bboxes:
        rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                bbox[3]-bbox[1],linewidth=1,edgecolor='g',facecolor='none')
        currentAxis.add_patch(rect)
        plt.axis('off') 

def showResTest(img_t,bboxes,bboxes_p,labels_p,score_p, score_thresh = 0.5,IoU_treshold=0.4, plot=True):
    if plot:
        fig = plt.figure()
        plt.imshow(img_t.cpu().numpy().transpose(1,2,0))
    currentAxis=plt.gca()
    
    if plot:
        for bbox,label,score in zip(bboxes_p,labels_p,score_p):
            if score > score_thresh:
                rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                       bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
                currentAxis.add_patch(rect)
                props = dict(boxstyle='round', facecolor='r', alpha=1) 
                plt.text(bbox[0], bbox[1],"%s: %.2f"%(validlabel[label],score),
                          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                # plt.text(bbox[0], bbox[1],"%s: %.2f"%(label,score),
                #          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                plt.axis('off')    
        return fig, bboxes_p_aligned,labels_p_aligned
    else:
        return _, bboxes_p_aligned,labels_p_aligned
    
def showRes(img_t,bboxes,bboxes_p,labels_p,score_p, score_thresh = 0.5,IoU_treshold=0.4, plot=True):
    if plot:
        fig = plt.figure()
        plt.imshow(img_t.cpu().numpy().transpose(1,2,0))
        currentAxis=plt.gca()
    
    # choose the most good prediction
    bboxes_p_aligned = []
    labels_p_aligned = []
    score_p_aligned = []
    for box_true in bboxes:
        IoU_t = []
        for box_p in bboxes_p:
            IoU_t.append(cal_IoU(box_true, box_p))
        validindex = np.array(IoU_t)>IoU_treshold
        # maxindex = IoU_t.index(max(IoU_t))
        scores = np.array(score_p)[validindex]
        if len(scores) >0:
            maxindex = score_p.index(np.max(scores))
            if IoU_t[maxindex] > IoU_treshold:
                bboxes_p_aligned.append(bboxes_p[maxindex])
                labels_p_aligned.append(labels_p[maxindex])          
                score_p_aligned.append(score_p[maxindex]) 
    
    if plot:
        for bbox,label,score in zip(bboxes_p_aligned,labels_p_aligned,score_p_aligned):
            if score > score_thresh:
                rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                       bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
                currentAxis.add_patch(rect)
                props = dict(boxstyle='round', facecolor='r', alpha=1) 
                plt.text(bbox[0], bbox[1],"%s: %.2f"%(validlabel[label],score),
                          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                # plt.text(bbox[0], bbox[1],"%s: %.2f"%(label,score),
                #          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                plt.axis('off')    
        return fig, bboxes_p_aligned,labels_p_aligned
    else:
        return _, bboxes_p_aligned,labels_p_aligned

def showRes_size(img_t,bbox,dia,length,dia_obs=None,len_obs=None,harvestEst=None,harvestObs=None, final_dia = None, final_len=None, final_yield = None):
    fig = plt.figure()
    img = img_t.cpu().numpy().transpose(1,2,0)
    
    if img.shape[0] > img.shape[1]:
        img = img.transpose(1,0,2)
        bbox = [bbox[1],bbox[0],bbox[3],bbox[2]]
    plt.imshow(img)    
    currentAxis=plt.gca()
                   
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    rect=patches.Rectangle((bbox[0], bbox[1]),w,
                           h,linewidth=1,edgecolor='r',facecolor='none')
    currentAxis.add_patch(rect)
    r = dia/2
    v = np.pi*(r**2) * length /1000 # cm^3
    freshweight = 0.51*v + 1.07 # g
    plt.text(bbox[0]-2, bbox[1],
             "Est. diameter: %.1f mm\nEst. length: %.1f mm\nEst. fresh weight: %.1f g\nEst. remaining days: %s\nPred. final D %s\nPred. final L %s"%(dia,length,freshweight,harvestEst,final_dia,final_len),
              bbox=dict(facecolor='r', edgecolor=(0, 0, 0, 0), pad=0),color='w',fontsize = 6)
    plt.text(bbox[0]-2, bbox[3]+1,
             "Obs. diameter: %s\nObs. length: %s\nObs. remaining days: %s"%(dia_obs,len_obs,harvestObs),
              bbox=dict(facecolor='g', edgecolor=(0, 0, 0, 0), pad=0),color='w',fontsize = 6,horizontalalignment='left',
              verticalalignment='top')
  
    plt.axis('off')    
    return fig

def showRes_size_ax(ax,img_t,bbox,dia,length,dia_obs=None,len_obs=None,harvestEst=None,
                    harvestObs=None, final_dia = None, final_len=None, final_yield = None):
  
    img = img_t.cpu().numpy().transpose(1,2,0)
    
    if img.shape[0] > img.shape[1]:
        img = img.transpose(1,0,2)
        bbox = [bbox[1],bbox[0],bbox[3],bbox[2]]
    ax.imshow(img)    
                   
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    rect=patches.Rectangle((bbox[0], bbox[1]),w,
                           h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    r = dia/2
    v = np.pi*(r**2) * length /1000 # cm^3
    freshweight = 0.51*v + 1.07 # g
    ax.text(bbox[0]-2, bbox[1],
             "Est. diameter: %.1f mm\nEst. length: %.1f mm\nEst. fresh weight: %.1f g\nEst. remaining days: %s\nPred. final D %s\nPred. final L %s"%(dia,length,freshweight,harvestEst,final_dia,final_len),
              bbox=dict(facecolor='r', edgecolor=(0, 0, 0, 0), pad=0),color='w',fontsize = 7)
    if final_yield is None:
        ax.text(bbox[0]-2, bbox[3]+1,
                 "Obs. diameter: %s mm\nObs. length: %s mm\nObs. remaining days: %s"%(dia_obs,len_obs,harvestObs),
                  bbox=dict(facecolor='g', edgecolor=(0, 0, 0, 0), pad=0),color='w',fontsize = 7,horizontalalignment='left',
                  verticalalignment='top')
    else:
        ax.text(bbox[0]-2, bbox[3]+1,
                 "Obs. diameter: %s mm\nObs. length: %s mm\nObs. remaining days: %s\nObs. fresh weight: %s g"%(dia_obs,len_obs,harvestObs,final_yield),
                  bbox=dict(facecolor='g', edgecolor=(0, 0, 0, 0), pad=0),color='w',fontsize = 7,horizontalalignment='left',
                  verticalalignment='top')  
  
    ax.axis('off')    

def cropImgByBbox(img, bboxes, obj_enlarge = 0.3, obj_size = 256, transforms=None):
    boxes = []      
    img_obj_list = []

    for boxes in bboxes:
        xmin = boxes[0]
        xmax = boxes[2]
        ymin = boxes[1]
        ymax = boxes[3]
      
        obj_width = xmax - xmin
        obj_height = ymax - ymin
        maxEdge = int(np.max([obj_width,obj_height]))
        
        cetral = (int((xmin+xmax)/2), int((ymin+ymax)/2))
   
        img_obj = img.crop(((cetral[0]-(1+obj_enlarge)*maxEdge/2),
                           (cetral[1]-(1+obj_enlarge)*maxEdge/2),
                           (cetral[0]+(1+obj_enlarge)*maxEdge/2),
                           (cetral[1]+(1+obj_enlarge)*maxEdge/2)))
        img_obj = img_obj.resize((obj_size, obj_size),resample = Image.Resampling.BILINEAR)
        if transforms is not None:
            img_obj,_ = transforms(img_obj)
        img_obj_list.append(img_obj)
        
    return img_obj_list

def paraEstimation(x_s, y_s, model_para,plot=True,ylabel='',number=''):
    xscale = 1/50
    resolution = 10
    para_scale = np.array([1/50, 1/500, -50])
    x0_list = np.arange(0,700,resolution).astype(np.float32)
    x_index = np.round((np.array(x_s))/resolution).astype(np.int64)
    tmp = np.zeros(len(x0_list))
    tmp[x_index] = np.array(y_s)*xscale
    para_in = torch.unsqueeze(torch.tensor(tmp.astype(np.float32)).to(device),dim=0)
    model_para.eval()
    cali_p = torch.squeeze(model_para(para_in)).detach().cpu().numpy()/para_scale
    
    if plot:
        fig,ax=plt.subplots(1,1)
        t=np.linspace(0, 600)
        ax.plot(t,util.func(t, *cali_p),'r-', label='Fitted Logistic curve')
        ax.scatter(np.array(x_s),np.array(y_s))       
        ax.set_xlabel('GDD')
        ax.set_ylabel(ylabel)
        fig.savefig('results/curve_fitting_%s_%s.png'%(ylabel,i), bbox_inches="tight")
        plt.close(fig)
    return cali_p

class demoTrajectory():
    def __init__(self):
        ids = 12 # the No. on the tag of the demo fruit
        obs_dates = ['07/26/2023', '07/29/2023', '08/01/2023', '08/04/2023', '08/07/2023', '08/11/2023']
        self.dates = [datetime.datetime.strptime(t,'%m/%d/%Y') for t in obs_dates]
        self.diameter = [8.05, 11.39, 11.42, 14.5, 18.45, 23.94]
        self.length = [10.09, 14.19, 15.72, 19.53, 25.3, 30.93]
        self.GDDlist = [0, 75.8755982300445, 141.46474267187557, 216.5649807375197, 283.1683296767341, 369.87646102173585]
        self.freshYield = '7.28' # for No.12 Jul. 26 - Aug.11
        imgRoot = 'datasets/strawberry_img_tagged'
        self.imgPathes = ['%s/%s_%s'%(imgRoot,t.strftime('%Y%m%d'),ids) for t in self.dates]
    def __len__(self):
        return len(self.dates)
    
if __name__ == '__main__':
    ## settings
    util.mkdir('results')
    
    ## load faster-rcnn
    model_fasterRcnn = torch.load('models/fasterRcnn.pth')
    model_fasterRcnn.to(device)
    
    ## load cnn
    ## crop img into the CNN
    modelName = 'densenet121'
    modelPath = 'models/densenet121_state_dict.pth'     
    model_cnn = to_device(util.ImageClassificationModel(model=modelName, outNum = 3), device)
    model_cnn.load_state_dict(torch.load(modelPath)) 
    
    ##  load parameter network
    model_para_dia = util.NN_para(input_dim=70, output_dim=3)
    modelName_dia = 'models/paraNet_state_dict_diameter.pth'
    model_para_dia.load_state_dict(torch.load(modelName_dia)) 
    model_para_dia.to(device)
    
    model_para_len = util.NN_para(input_dim=70, output_dim=3)
    modelName_len = 'models/paraNet_state_dict_length.pth'
    model_para_len.load_state_dict(torch.load(modelName_len)) 
    model_para_len.to(device)
    
    # load growth networks
    model_grow_dia = util.NN_growth(input_dim=5, output_dim=1)
    modelName_dia_grow = 'models/growNet_state_dict_diameter.pth'      
    model_grow_dia.load_state_dict(torch.load(modelName_dia_grow))  
    model_grow_dia.to(device)
    model_grow_dia.eval()
    
    model_grow_len = util.NN_growth(input_dim=5, output_dim=1)
    modelName_len_grow = 'models/growNet_state_dict_length.pth'      
    model_grow_len.load_state_dict(torch.load(modelName_dia_grow))  
    model_grow_len.to(device)
    model_grow_len.eval()
    scale_growNet=1/50
    
    # load demo fruit
    demo = demoTrajectory()
  
    normlize_coef=1/40
    scale = 1/4
    validlabel = ['background','flower','small g','green','white','turning red','red','overripe']
    trans = util.ToTensor()
    
    # reference curve
    traj = util.getTrajectories()
    popt, _, _,_,_ = traj.fitBoundary(mode='diameter',plot=False)
    valid_traj_list, complete_traj_list, meanlifespan_list = traj.retrieveHighQualityTraj()
    
    i=0
    currentGDDList= []  # GDD from small g stage
    diaList,lengthList,phenofList = [],[],[]
    
    fig, axs = plt.subplots(3,2,figsize = (7.8,9))
    axx = axs.flatten(order='C')
    numList = ['(a)','(b)','(c)','(d)','(e)','(f)']
    # for p,d,g,ax,no in zip(demo.imgPathes, demo.dates, demo.GDDlist, axx, numList):
    for i in range(len(demo)):
        ax=axx[i]
        img = Image.open('%s.jpg'%demo.imgPathes[i]).convert("RGB")
        size_0 = img.size
        img = ImageOps.exif_transpose(img)   # rotating the img when the up direction saved in exif
        size_1 = img.size
        if size_0!=size_1:
            print("Shape of img {} is opposited, corrected from {} to {}".format(demo.imgPathes[i],size_0,size_1))
            
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)),resample=0)
        img_t,_ = trans(img)
        
        # load real bounding box
        # with 0 being background
        with open('%s.json'%demo.imgPathes[i],'r') as f:
            labeldata= json.load(f)
        num_objs = len(labeldata['shapes'])
        bboxes = []      
 
        for label in labeldata['shapes']:
            xmin = label['points'][0][0] *scale
            xmax = label['points'][1][0] *scale
            ymin = label['points'][0][1] *scale
            ymax = label['points'][1][1] *scale
            bboxes.append([xmin, ymin, xmax, ymax])
        model_fasterRcnn.eval()       
        # predicted
        predicted = model_fasterRcnn(img_t.unsqueeze(0).to(device))  
        bboxes_p = predicted[0]['boxes'].cpu().tolist()
        labels_p = predicted[0]['labels'].cpu().tolist()
        score_p = predicted[0]['scores'].cpu().tolist()
        
        # show img bounding box
        _,bboxes_p_aligned,labels_p_aligned = showRes(img_t,bboxes,bboxes_p,labels_p,score_p,plot=False)
        
        # estimate the fruit size
        img_obj_list = cropImgByBbox(img, bboxes_p_aligned, obj_enlarge = 0.2, obj_size = 256, transforms=trans)
        
        if len(img_obj_list)>0:
            t = img_obj_list[0]
            model_cnn.eval()
            res_tmp = torch.squeeze(model_cnn(t.unsqueeze(0).to(device)))
            dia_p = (res_tmp[0]/normlize_coef).cpu().detach().numpy()
            len_p = (res_tmp[1]/normlize_coef).cpu().detach().numpy()
            phenof_p = (res_tmp[2]/0.1).cpu().detach().numpy()
            
            dia_obs = '%.1f'%demo.diameter[i] 
            len_obs = '%.1f'%demo.length[i]
                         
            diaList.append(dia_p)
            lengthList.append(len_p)
            phenofList.append(phenof_p)
            plt.axis('off') 
            
            # mean daily GDD for the previous seven days
            timeSpan = [(demo.dates[i]-datetime.timedelta(7)).strftime('%m/%d/%Y'),demo.dates[i].strftime('%m/%d/%Y')]
            tmp = traj.weather.iloc[traj.weather[traj.weather['Date']==timeSpan[0]].index[0]:
                                traj.weather[traj.weather['Date']==timeSpan[1]].index[0]]   
            mean_dailyGDD=np.mean(tmp['GDD'])
            
            if len(labels_p_aligned)==0:
                fig = showRes_size(img_t,bboxes_p_aligned[0],dia_p,len_p)
                
            else:            
                if i==0:                    
                    currentGDD_p = util.func_reverse(dia_p,*popt)
                    currentGDD = max(currentGDD_p,0)
    
                else:
                    currentGDD = currentGDDList[0] + demo.GDDlist[i]
                currentGDDList.append(currentGDD)
                    
                # estimation the logistic function parameters
                cali_p_dia = paraEstimation(x_s=currentGDDList, y_s=diaList, model_para=model_para_dia,ylabel='Diameter mm',plot=True,number=i)        
                cali_p_len = paraEstimation(x_s=currentGDDList, y_s=lengthList, model_para=model_para_len,ylabel='Length mm',plot=True,number=i)   
                
                remainedGDD,remainedDay = util.calHavestTime_continuous(mean_dailyGDD,currentGDD,pheno=phenof_p,plot=False, meanlifespan=meanlifespan_list[-1])
                remainedDay = '%.1f days'%remainedDay
                remainedDayObs = (demo.dates[-1] - demo.dates[i]).days
                remainedDayObs = '%s days'%remainedDayObs                
                
                # directly predict final size using laterest obs as startpoint               
                inTensor = torch.tensor(np.array([float(dia_p),remainedGDD,*cali_p_dia]).astype(np.float32)).to(device).unsqueeze(dim=0)
                final_dia = max(model_grow_dia(inTensor).item()/scale_growNet,float(dia_p))
                final_dia = '%.1f mm'%(final_dia)
                
                inTensor = torch.tensor(np.array([float(len_p),remainedGDD,*cali_p_len]).astype(np.float32)).to(device).unsqueeze(dim=0)
                final_len = max(model_grow_len(inTensor).item()/scale_growNet,float(len_p))
                final_len = '%.1f mm'%(final_len)                        
                
                if i==5:
                    showRes_size_ax(ax,img_t,bboxes_p_aligned[0],dia_p,len_p,dia_obs,len_obs,
                                       harvestEst=remainedDay,harvestObs=remainedDayObs,final_dia=final_dia, final_len=final_len, final_yield=demo.freshYield)
                else:
                    showRes_size_ax(ax,img_t,bboxes_p_aligned[0],dia_p,len_p,dia_obs,len_obs,
                                       harvestEst=remainedDay,harvestObs=remainedDayObs,final_dia=final_dia, final_len=final_len)
                # Set the background color
                background_color = 'white'
                
                # Add the text with a white background
                text = numList[i]
                text_x, text_y = 50, 50  # Adjust the text position as needed
                
                # Create text with white background
                ax.text(text_x, text_y, text,
                        fontsize=18,  # Adjust the font size as needed
                        ha='center', va='center',  # Text alignment
                        color='black',  # Text color
                        bbox=dict(boxstyle='square', facecolor=background_color, edgecolor='none'))  # Set the background color


    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    fig.savefig('results/KGCV_demo.png', bbox_inches="tight")  