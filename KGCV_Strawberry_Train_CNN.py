# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:42:55 2024

@author: yang8460

Train the CNN for fruit size & decimal phenological stage estimation
"""
import KGCV_util as util
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import matplotlib
import datetime
import sys
import time
import glob
import json
import random
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
    def __init__(self, root, transforms = None, scale = 1/4, obj_enlarge = 0.3, obj_size = 256, normlize_coef=1/40):
        self.root = root
        self.transforms = transforms
        self.imgs = glob.glob('%s/*.jpg'%root)
        self.labels = ['%s.json'%(t.split('.')[0]) for t in self.imgs]
        self.validlabel = ['flower','small g','green','white','turning red','red','overripe']
        self.scale = scale
        self.obj_enlarge = obj_enlarge
        self.obj_size = obj_size
        self.normlize_coef = normlize_coef
        self.normlize_coef_pheno = 0.1
        
    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((int(img.size[0]*self.scale), int(img.size[1]*self.scale)),resample=Image.Resampling.BILINEAR)
        img = ImageOps.exif_transpose(img)   # rotating the img when the up direction saved in exif

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        with open(label_path,'r') as f:
            labeldata= json.load(f)
           
        # labels = list(np.loadtxt(label_path))
        num_objs = len(labeldata['shapes'])
        boxes = []      
        labels_diameter = []
        labels_len = []
        img_obj_list = []
        labels_pheno_base = []
        labels_pheno_sub = []
        labels_pheno_percent = []
        for label in labeldata['shapes']:
            xloc = [label['points'][0][0],label['points'][1][0]]
            xmin = np.min(xloc) *self.scale   # doing this is because sometime the labelme will flip the bounding box
            xmax = np.max(xloc) *self.scale
            yloc = [label['points'][0][1],label['points'][1][1]]
            ymin = np.min(yloc) *self.scale
            ymax = np.max(yloc) *self.scale
            boxes.append([xmin, ymin, xmax, ymax])
            
            labels_pheno_base.append(self.validlabel.index(label['label'].split(', ')[0])) # 0-6, no background
            labels_pheno_sub.append(np.float32(label['label'].split(', ')[3])) # 0-1
            
            labels_pheno_percent.append(labels_pheno_base[-1]+labels_pheno_sub[-1])
            labels_diameter.append(np.float32(label['label'].split(', ')[1]))
            labels_len.append(np.float32(label['label'].split(', ')[2]))
            obj_width = xmax - xmin
            obj_height = ymax - ymin
            maxEdge = int(np.max([obj_width,obj_height]))
            
            cetral = (int((xmin+xmax)/2), int((ymin+ymax)/2))
            img_obj = img.crop(((cetral[0]-(1+self.obj_enlarge)*maxEdge/2),
                               (cetral[1]-(1+self.obj_enlarge)*maxEdge/2),
                               (cetral[0]+(1+self.obj_enlarge)*maxEdge/2),
                               (cetral[1]+(1+self.obj_enlarge)*maxEdge/2)))
            img_obj = img_obj.resize((self.obj_size, self.obj_size),resample = Image.Resampling.BILINEAR)
            if self.transforms is not None:
                img_obj = self.transforms(img_obj)
            img_obj_list.append(img_obj)
            
        labels_diameter = torch.as_tensor(labels_diameter, dtype=torch.float32)*self.normlize_coef
        labels_len = torch.as_tensor(labels_len, dtype=torch.float32)*self.normlize_coef 
        labels_pheno_percent = torch.as_tensor(labels_pheno_percent, dtype=torch.float32)*self.normlize_coef_pheno
        
        obs_num_list = [t for t in range(num_objs)]
        loc = random.choice(obs_num_list)
        return img_obj_list[loc], [labels_diameter[loc],labels_len[loc],labels_pheno_percent[loc]]

    def __len__(self):
        return len(self.imgs)

class DatasetMultiobs(object):
    def __init__(self, root, transforms = None, scale = 1/4, obj_enlarge = 0.3, obj_size = 256, normlize_coef=1/40):
        self.root = root
        self.transforms = transforms
        self.imgs = glob.glob('%s/*.jpg'%root)
        self.labels = ['%s.json'%(t.split('.')[0]) for t in self.imgs]
        self.validlabel = ['flower','small g','green','white','turning red','red','overripe']
        self.scale = scale
        self.obj_enlarge = obj_enlarge
        self.obj_size = obj_size
        self.normlize_coef = normlize_coef
        self.normlize_coef_pheno = 0.1
        
    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((int(img.size[0]*self.scale), int(img.size[1]*self.scale)),resample=Image.Resampling.BILINEAR)
        img = ImageOps.exif_transpose(img)   # rotating the img when the up direction saved in exif

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        with open(label_path,'r') as f:
            labeldata= json.load(f)
           
        boxes = []      
        labels_diameter = []
        labels_len = []
        img_obj_list = []
        labels_pheno_base = []
        labels_pheno_sub = []
        labels_pheno_percent = []
        for label in labeldata['shapes']:
            xloc = [label['points'][0][0],label['points'][1][0]]
            xmin = np.min(xloc) *self.scale   # doing this is because sometime the labelme will flip the bounding box
            xmax = np.max(xloc) *self.scale
            yloc = [label['points'][0][1],label['points'][1][1]]
            ymin = np.min(yloc) *self.scale
            ymax = np.max(yloc) *self.scale
            boxes.append([xmin, ymin, xmax, ymax])
            
            labels_pheno_base.append(self.validlabel.index(label['label'].split(', ')[0])) # 0-6, no background
            labels_pheno_sub.append(np.float32(label['label'].split(', ')[3])) # 0-1
            
            labels_pheno_percent.append(labels_pheno_base[-1]+labels_pheno_sub[-1])
            labels_diameter.append(np.float32(label['label'].split(', ')[1]))
            labels_len.append(np.float32(label['label'].split(', ')[2]))
            obj_width = xmax - xmin
            obj_height = ymax - ymin
            maxEdge = int(np.max([obj_width,obj_height]))
            
            cetral = (int((xmin+xmax)/2), int((ymin+ymax)/2))
            img_obj = img.crop(((cetral[0]-(1+self.obj_enlarge)*maxEdge/2),
                               (cetral[1]-(1+self.obj_enlarge)*maxEdge/2),
                               (cetral[0]+(1+self.obj_enlarge)*maxEdge/2),
                               (cetral[1]+(1+self.obj_enlarge)*maxEdge/2)))
            img_obj = img_obj.resize((self.obj_size, self.obj_size),resample = Image.Resampling.BILINEAR)
            if self.transforms is not None:
                img_obj = self.transforms(img_obj)
            img_obj_list.append(img_obj)
            
        labels_diameter = torch.as_tensor(labels_diameter, dtype=torch.float32)*self.normlize_coef
        labels_len = torch.as_tensor(labels_len, dtype=torch.float32)*self.normlize_coef 
        labels_pheno_percent = torch.as_tensor(labels_pheno_percent, dtype=torch.float32)*self.normlize_coef_pheno
        
        return img_obj_list, [labels_diameter,labels_len,labels_pheno_percent]

    def __len__(self):
        return len(self.imgs)
    
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

def get_transform(train):
   
    if train:       
        transformer = torchvision.transforms.Compose(
            [  # Applying Augmentation
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomRotation(20),
                # torchvision.transforms.RandomAffine(degrees=20,scale=(0.9,1.2)),
                torchvision.transforms.ColorJitter(brightness=((0.7,1.3)),saturation=((0.7,1.3))),
                torchvision.transforms.ToTensor(),
     
            ]
        )
    else:
        transformer = torchvision.transforms.Compose(
            [  # Applying Augmentation
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
            ]
        )    
    return transformer

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def show_example(img, label=''):
    plt.figure()
    print('Label ',label)
    plt.imshow(img.permute(1, 2, 0))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def test(model, test_loader):
    model.eval()
    outputs = [model.test_step(batch) for batch in test_loader]

    outputs_vector = torch.concat(outputs, dim=0)
    return outputs_vector.detach().cpu().numpy()

def layerExtract(my_list,layer):
    r = False
    for t in my_list:
        if t in layer:
            r = True
            break
    return r

def fit(epochs, lr, lr_finetune_coef,lr_decay, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    # my_list = ['network.fc.weight', 'network.fc.bias',
    #            'network.classifier.weight', 'network.classifier.bias']
    # params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    # base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))
    my_list = ['fc','bn','classifier','norm']
    params = [t for t in model.named_parameters() if layerExtract(my_list,t[0])]
    base_params = [t for t in model.named_parameters() if not layerExtract(my_list,t[0])]
    print('The fc params: len %d with lr = %.6f'%(len(params),lr))
    print('The base params: len %d with lr = %.6f'%(len(base_params),lr*lr_finetune_coef))
    trainParams = [
        {"params": [t[1] for t in params], "lr": lr},
        {"params": [t[1] for t in base_params], "lr": lr*lr_finetune_coef},
    ]
    optimizer = opt_func(trainParams)
    # Decay LR by a factor every epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        finishTime = time.time()
        for i,batch in enumerate(train_loader):
            startTime = time.time()
            # print('takes %.2f s to load batch'%(startTime-finishTime))
            loss = model.training_step(batch)
            lossTotal = loss
            train_losses.append(loss)
            lossTotal.backward()
            optimizer.step()
            optimizer.zero_grad()
            finishTime = time.time()
            # print('takes %.2f s to train'%(finishTime - startTime))
            
        lrList = [param_group['lr'] for param_group in optimizer.param_groups]

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
      
        model.epoch_end(epoch, result,lrList)
        history.append(result)
        
        # lr decay
        exp_lr_scheduler.step()
        
    return history

def plot_accuracies(history,outFolder,title='acc',saveFig=False):
    fig=plt.figure()
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. epochs')
    if saveFig:    
        fig.savefig('%s/acc.png'%outFolder)
    
def plot_loss(history,outFolder,title='loss',saveFig=False):
    fig=plt.figure()
    loss_train_base = [x['train_loss'] for x in history]
    loss_val_base = [x['val_loss'] for x in history]
    plt.plot(loss_train_base, 'r-',label = 'Train base')
    plt.plot(loss_val_base, 'y-',label = 'Test base')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss vs. epochs')
    if saveFig:  
        fig.savefig('%s/loss.png'%outFolder)

def plot_R2(history,outFolder,title='SlopeAndR2',saveFig=False):
    fig=plt.figure()
   
    val_R2_1 = [x['R2_dia'] for x in history]   
    val_R2_2 = [x['R2_len'] for x in history]
    val_R2_3 = [x['R2_pheno'] for x in history] 
    plt.plot(val_R2_1, 'y--',label = 'val_R2_dia')
    plt.plot(val_R2_2, 'r--',label = 'val_R2_len')
    plt.plot(val_R2_3, 'b--',label = 'val_R2_pheno')
    plt.xlabel('epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.title('R2 vs. epochs')
    if saveFig:    
        fig.savefig('%s/SlopeAndR2.png'%outFolder)  

def obsLabel(test_dl):
    labelList1 = []
    labelList2 = []
    labelList3 = []
    for _, labels in test_dl:
        label1, label2, label3 = labels
        labelList1.append(label1.cpu().numpy())
        labelList2.append(label2.cpu().numpy())
        labelList3.append(label3.cpu().numpy())
    labelList1_cat = np.concatenate(labelList1,axis=1)
    labelList2_cat = np.concatenate(labelList2,axis=1)
    labelList3_cat = np.concatenate(labelList3,axis=1)
    return np.squeeze(labelList1_cat), np.squeeze(labelList2_cat), np.squeeze(labelList3_cat)

def show_estimation(x,y,title='',LimRange=None):
    x=np.array(x)
    y=np.array(y)
    x,y = util.removeNaN(obs=x, pre=y)
    fig, ax = plt.subplots(1, 1,figsize = (6,5))
    R2 = []
    plt.scatter(x,y,color = 'b')

    if len(y) > 1:
        para = np.polyfit(x, y, 1)
        t=[np.min(x),np.max(x)]
        y_fit = np.polyval(para, t)  #
        ax.plot(t, y_fit,
                color = 'r', linestyle = '--',dashes=(5, 5),label='fitted curve')
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
        # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
        ax.text(0.1, 0.83, r'$R^2 $= ' + str(R2)[:5], transform=ax.transAxes,fontsize=14)
        # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
        ax.text(0.1, 0.76 , r'$RMSE $= ' + str(RMSE)[:5], transform=ax.transAxes, fontsize=14)
    
        ax.text(0.1,  0.69, r'$Slope $= ' + str(para[0])[:5], transform=ax.transAxes, fontsize=14)
    if LimRange != None:
        plt.xlim(LimRange)
        plt.ylim(LimRange)
        ax.plot(LimRange, LimRange, 'k', label='1:1 line')
    else:
        ax.plot([np.min(x),np.max(x)], [np.min(x),np.max(x)], 'k', label='1:1 line')
    ax.legend(loc=1,edgecolor = 'w',facecolor='w', framealpha=1, ncol = 2)
    plt.xlabel('Observed', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.title(title)
    plt.legend(loc = 4)
    return fig

def checkLabel(labels):
    valid = True
    for t in labels:
        if not len(t.split(', ')) == 4:
            valid = False
            
    return valid

def checkDataset():
    ## check label valibility
    folders = ['datasets/strawberry_fruitSize_v2/train','datasets/strawberry_fruitSize_v2/test']
    unvalidsample = []
    for folder in folders:
        jsonList = glob.glob('%s/*.json'%folder)
        
        # load json
        for j in jsonList:
            with open(j,'r') as f:
                json_tmp = json.load(f)
                
            labels = [t['label'] for t in json_tmp['shapes']]
            valid = checkLabel(labels)
            if not valid:
                unvalidsample.append(j)
                
        print('processing %s'%folder)
           
if __name__ == "__main__":
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    saveResult = True
    num_epochs = 100
    opt_func = torch.optim.Adam
    lr_finetune_coef =  0.01 # 0.1
    lr = 0.005
    lr_decay = 0.98
    batch_size = 32

    criterion = nn.MSELoss()
    outNum = 3
    
    # image patches extend coef
    obj_enlarge = 0.2
    
    # model informaltion
    modelName = 'densenet121'
    projectName = '%s-epoch%d-batch%d_lr%s_ftcoef%s_enlarge%s'%(modelName,num_epochs,batch_size,lr,lr_finetune_coef,obj_enlarge)
    outFolder = 'log/%s-%s'%(projectName,now)
               
    # load dataset
    train_ds = Dataset(root = 'datasets/strawberry_fruitSize/train', transforms = get_transform(train=True),obj_enlarge = obj_enlarge)
    val_ds = Dataset(root = 'datasets/strawberry_fruitSize/test', transforms = get_transform(train=False),obj_enlarge = obj_enlarge)
    test_ds = DatasetMultiobs(root = 'datasets/strawberry_fruitSize/test', transforms = get_transform(train=False),obj_enlarge = obj_enlarge)    
    print('Train set: %d, vali set: %d'%(len(train_ds), len(val_ds)))
    
    # make dataloader
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True, num_workers=3)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size = 1, shuffle=False, num_workers=0) 

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)    
    
    # load model
    model = to_device(util.ImageClassificationModel(model=modelName, criterion = criterion, outNum = outNum), device)
    
    # model test
    tmp = evaluate(model, val_dl)
    print(tmp)

    # training
    history = fit(num_epochs, lr,lr_finetune_coef,lr_decay, model, train_dl, val_dl, opt_func)
    
    # save results
    if saveResult:
        
        if not os.path.exists('log'):
            os.mkdir('log') 
        if not os.path.exists(outFolder):
            os.mkdir(outFolder)
        outPath = 'models'
        if not os.path.exists(outPath):
            os.mkdir(outPath)
            
    # plot training loss and R2
    plot_loss(history,outFolder=outFolder,title=projectName,saveFig=saveResult)
    plot_R2(history,outFolder=outFolder,title=projectName,saveFig=saveResult)

    # test model   
    out = test(model=model, test_loader=test_dl)
    out = np.array(out)
    
    # # plot test random
    labelList1, labelList2, labelList3 = obsLabel(test_dl)
    
    # # show eachClass
    fig1 = show_estimation(x=labelList1/train_ds.normlize_coef, 
                          y =out[:,0]/train_ds.normlize_coef ,title='',LimRange=None)
    fig2 = show_estimation(x=labelList2/train_ds.normlize_coef, 
                          y = out[:,1]/train_ds.normlize_coef,title='',LimRange=None)
    fig3 = show_estimation(x=labelList3/train_ds.normlize_coef_pheno, 
                          y = out[:,2]/train_ds.normlize_coef_pheno,title='',LimRange=None)
   
    if saveResult:
        fig1.savefig('%s/scatter_dia.png'%outFolder)
        fig2.savefig('%s/scatter_len.png'%outFolder)
        fig3.savefig('%s/scatter_pheno.png'%outFolder)
        save_object(history, '%s/history.pkl'%outFolder)
    # save model
    if saveResult:
        torch.save(model.state_dict(), '%s/%s-%s_state_dict.pth'%(outPath,projectName,now))
        torch.save(model, '%s/%s-%s.pth'%(outPath,projectName,now))
