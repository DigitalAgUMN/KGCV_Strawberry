# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:39:01 2024

@author: yang8460

Train Faster-RCNN for strawberries detection
"""

import os
import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import math
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
sys.path.append(r'faster_rcnn/detection')
import utils
import transforms as T
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import time
import glob
import json
import random
import datetime
version = int(sys.version.split()[0].split('.')[1])
if version > 7:
    import pickle
else:
    import pickle5 as pickle
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        data = pickle.load(inp)
    return data
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

class Dataset(object):
    def __init__(self, root, transforms = None, scale = 1/4):
        self.root = root
        self.transforms = transforms
        self.imgs = glob.glob('%s/*.jpg'%root)
        self.labels = ['%s.json'%(t.split('.')[0]) for t in self.imgs]
        self.validlabel = ['background','flower','small g','green','white','turning red','red','overripe']
        self.scale = scale
        
    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((int(img.size[0]*self.scale), int(img.size[1]*self.scale)),resample=0)
        size_0 = img.size
        img = ImageOps.exif_transpose(img)   # rotating the img when the up direction saved in exif
        size_1 = img.size
        if size_0!=size_1:
            print("Shape of img {} is opposited, corrected from {} to {}".format(img_path,size_0,size_1))
        # print('test')
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        with open(label_path,'r') as f:
            labeldata= json.load(f)
           
        # labels = list(np.loadtxt(label_path))
        num_objs = len(labeldata['shapes'])
        boxes = []      
        labels = []
        for label in labeldata['shapes']:
            xloc = [label['points'][0][0],label['points'][1][0]]
            xmin = np.min(xloc) *self.scale   # doing this is because sometime the labelme will flip the bounding box
            xmax = np.max(xloc) *self.scale
            yloc = [label['points'][0][1],label['points'][1][1]]
            ymin = np.min(yloc) *self.scale
            ymax = np.max(yloc) *self.scale
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.validlabel.index(label['label'].split(', ')[0])) # label = 0 is background
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # debug
        for ts in targets:
            for t in ts['boxes']:
                a = t[3].cpu().numpy()-t[1].cpu().numpy() 
                if a<=0:
                    print(ts)
                
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor()) 
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort(contrast=(0.8, 1.2),
         saturation=(0.8, 1.2),
         brightness=(0.8, 1.2),p=1))
       
    return T.Compose(transforms)

def plot_accuracies(log_test,outFolder='log',title='Accuracy vs. epochs',saveFig=False):
    fig=plt.figure()
    AP50_95 = [x[0] for x in log_test]
    AP50 = [x[1] for x in log_test]
    AR10 = [x[7] for x in log_test]
    plt.plot(AP50_95, 'r-',label = 'AP IoU 0.5:0.95')
    plt.plot(AP50, 'g-',label = 'AP IoU 0.5')
    plt.plot(AR10, 'y-',label = 'AR max detection 10')
    plt.xlabel('epoch')
    plt.ylabel('AP & AR')
    plt.legend()
    plt.title(title)
    if saveFig:    
        fig.savefig('%s/AP.png'%outFolder)
    
def plot_loss(log_train,outFolder='log',title='Loss vs. epochs',saveFig=False):
    fig=plt.figure()
    loss_base = [dict(x)['loss'] for x in log_train]
    loss_classifier = [dict(x)['loss_classifier'] for x in log_train]
    loss_box_reg = [dict(x)['loss_box_reg'] for x in log_train]
    loss_objectness = [dict(x)['loss_objectness'] for x in log_train]
    loss_rpn_box_reg = [dict(x)['loss_rpn_box_reg'] for x in log_train]
    plt.plot(loss_base, 'r-',label = 'loss')
    plt.plot(loss_classifier, 'g-',label = 'loss_classifier')
    plt.plot(loss_box_reg, 'y-',label = 'loss_box_reg')
    plt.plot(loss_objectness, 'b-',label = 'loss_objectness')
    plt.plot(loss_rpn_box_reg, 'c-',label = 'loss_rpn_box_reg')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)
    if saveFig:  
        fig.savefig('%s/loss.png'%outFolder)

def datasetCheck(data_loader):
    bboxesList = []
    areaList = []
    labelList = []
    idList = []
    i=0
    for imgs, targets in data_loader:
        for img, target in zip(imgs, targets):
            for bbox,area,l,img_id in zip(target['boxes'].cpu().tolist(),target['area'].cpu().tolist(), 
                                          target['labels'].cpu().tolist(),target['image_id'].cpu().tolist()):
                bboxesList.append(bbox)
                areaList.append(area)
                labelList.append(l)
                idList.append(img_id)
        if i%30==0:
            print('processed %d batches'%i)
        i+=1

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        print ('set cudnn seed')
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        print ('set cuda seed')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def checkDataloader(data_loader):
    for img, target in data_loader:
        print('img.shape:', img[0].cpu().numpy().shape)
        bboxes = target[0]['boxes'].cpu().tolist()
        print('target.boxes:{},area:{},labels:{},id:{}'.format(bboxes, 
              target[0]['area'], target[0]['labels'],
               target[0]['image_id']))
        plt.figure()
        plt.imshow(img[0].cpu().numpy().transpose(1,2,0))
        currentAxis=plt.gca()
        for bbox in bboxes:
            rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                   bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
            currentAxis.add_patch(rect)
        break
    
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # settings
    saveResult = True
    num_classes = 7 + 1 # classes plus background
    print_freq = 100
    num_epochs = 50
    seed=1
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # make dataloader
    note = 'Faster_RCNN'        
    dataset = Dataset(root = 'datasets/strawberry_v5/train', transforms = get_transform(train=True))
    dataset_test = Dataset(root = 'datasets/strawberry_v5/test', transforms = get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(
       dataset, batch_size=2, shuffle=True, num_workers=0,
       collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # check dataloader
    checkDataloader(data_loader)
    
    # get the model
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.8, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.005)
                                
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=1,
                                                   gamma=0.95)

    # let's train it
    log_train = []
    log_test = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
        log_train.append([(t[0], t[1].avg) for t in metric_logger.meters.items()])
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(model, data_loader_test, device=device)
        log_test.append(evaluator.coco_eval['bbox'].stats)
        
    # save model
    outPath = 'models'
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    if not os.path.exists('log'):
        os.mkdir('log')
    
    if saveResult:
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        logPath = '%s_%s_epoch%d'%(note,now,num_epochs)
        torch.save(model, '%s/fasterRcnn-%s.pth'%(outPath,logPath))
        torch.save(model.state_dict(), '%s/fasterRcnn-%s_state_dict.pth'%(outPath,logPath))
        
        if not os.path.exists('log/%s'%logPath):
            os.mkdir('log/%s'%logPath)
        plot_accuracies(log_test,outFolder='log/%s'%logPath,saveFig=saveResult)
        plot_loss(log_train,outFolder='log/%s'%logPath,saveFig=saveResult)
        obj = {}
        obj['log_train'] = log_train
        obj['log_test'] = log_test
        save_object(obj, 'log/%s/log.pkl'%logPath)
        
    # test for n samples
    n = 4
    labelName = dataset.validlabel
    i = 0
    for img, target in data_loader_test:
        model.eval()
        predicted = model(img[0].unsqueeze(0).to(device))
        bboxes_p = predicted[0]['boxes'].cpu().tolist()
        bboxes = target[0]['boxes'].cpu().tolist()
        labels = target[0]['labels'].cpu().tolist()
        labels_p = predicted[0]['labels'].cpu().tolist()
        score_p = predicted[0]['scores'].cpu().tolist()

        plt.figure()
        plt.imshow(img[0].cpu().numpy().transpose(1,2,0))
        currentAxis=plt.gca()
        for bbox,label in zip(bboxes,labels):
            rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                   bbox[3]-bbox[1],linewidth=1,edgecolor='g',facecolor='none')
            currentAxis.add_patch(rect)
            
        for bbox,label,score in zip(bboxes_p,labels_p,score_p):
            if score > 0.5:
                rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                       bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
                currentAxis.add_patch(rect)
                props = dict(boxstyle='round', facecolor='r', alpha=1) 
                plt.text(bbox[0], bbox[1],"%s: %.2f"%(labelName[label],score),
                          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                # plt.text(bbox[0], bbox[1],"%s: %.2f"%(label,score),
                #          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                plt.axis('off')
                
        i += 1
        if i >= n:
            break