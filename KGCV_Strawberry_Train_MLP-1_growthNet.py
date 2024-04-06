# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 23:42:31 2024

@author: yang8460

Train a MLP for fruit growth modeling

Establish the mapping l_t2 = f(l_t1,delta_GDD,Para), 
l is diameter ot length at time t; Para is the curve parameters
"""
import KGCV_util as util
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import os
import random

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

class MyDataset(Dataset):
    def __init__(self, X,y,para=None,inputMode='GDD',test=False):
        self.X=X
        self.y=y
        self.inputMode=inputMode
        self.test=test
        self.para=para
        self.paraDefault = [22.51785856354884,210.81902810721968,-0.011702128027145604]
        
    def __getitem__(self, index):
        y = self.y[index]
        x = np.array(self.X[index])
        if not self.test:
            noise = np.random.randn(len(x))*x*0.02
            x+=noise
        if self.inputMode=='GDD':
            xOut = x[:2]
        elif self.inputMode=='GDD_RAD':
            xOut = x
        else:
            raise ValueError
        if self.para is None:          
            xOut = list(xOut)
            xOut.extend(self.paraDefault)
        elif len(self.para)==1:
            xOut = list(xOut)
            xOut.extend(self.para[0])
        else:
            p = self.para[index]
            xOut = list(xOut)
            xOut.extend(p)
        return np.array(xOut).astype(np.float32),np.array(y).astype(np.float32)
    
    def __len__(self):
        return len(self.X)

def listSplit(l,ref_index):
    inList = []
    outList = []
    for i,t in enumerate(l):
        if i in ref_index:
            inList.append(t)
        else:
            outList.append(t)
    return inList,outList

def train_test_split(X, y, test_ratio=0.1):
    np.random.seed(0)
    indexList =np.arange(0,len(X))
    np.random.shuffle(indexList)
    
    # split train and test with no leak strategy
    indexTest = indexList[:int(test_ratio*len(X))]
    X_test, X_train = listSplit(X,indexTest)
    y_test, y_train = listSplit(y,indexTest)
    
    return X_train, X_test, y_train, y_test

def listSplit_no_shuffle(v,loc):

    return v[:loc],v[loc:]

def train_test_split_no_shuffle(X, y, z, test_ratio=0.1):
    np.random.seed(0)
    split_loc = int(test_ratio*len(X))

    X_test, X_train = listSplit_no_shuffle(X,split_loc)
    y_test, y_train = listSplit_no_shuffle(y,split_loc)
    z_test, z_train = listSplit_no_shuffle(z,split_loc)
    
    return X_train, X_test, y_train, y_test, z_train, z_test

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.criterion = nn.MSELoss()
        self.input_dim=input_dim
        self.output_dim=output_dim
        
        # Fully connected layer
        self.fc_1 = nn.Linear(input_dim, 32)
        self.fc_2 = nn.Linear(32, 32)
        self.fc_3 = nn.Linear(32, 16)
        self.fc_4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        
        # Bn of inputs
        self.bn = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        x = self.bn(x)
        out = self.relu(self.fc_1(x))
        out = self.relu(self.fc_2(out))
        out = self.relu(self.fc_3(out))
        out = self.fc_4(out)
        
        # return out
        return torch.squeeze(out,dim=1)

    def training_step(self, batch):
        X,y = batch 
        out = self(X)                  # Generate predictions
        loss = self.criterion(y, out)
        return loss
    
    def validation_step(self, batch):
        X,y = batch 
        out = self(X)                   # Generate predictions
        loss = self.criterion(y, out)
        if self.output_dim == 1:  # for regression
            return {'val_loss': loss.detach(), 
                    'oriOut': out, 'oriLabels':y}
        else:
            raise ValueError('Situation not available.')

    def test_step(self, batch):
        X,y = batch 
        out = self(X)                    # Generate predictions
        return out
    
    def validation_epoch_end(self, outputs):

        if self.output_dim == 1:  # for regression
            epochOut = [x['oriOut'] for x in outputs]
            epochLabels = [x['oriLabels'] for x in outputs]
            outVec = []
            labelsVec = []
            batch_losses_base = [x['val_loss'] for x in outputs]
            epoch_loss_base = torch.stack(batch_losses_base).mean()   # Combine losses
           
            for t,k in zip(epochOut,epochLabels):
                outVec.extend(t.cpu().numpy())
                labelsVec.extend(k.cpu().numpy())
            R2 = np.corrcoef(np.array(outVec), np.array(labelsVec))[0,1]**2
            R2 = torch.from_numpy(np.array(R2))
            return {'val_loss': epoch_loss_base.item(),
                    'R2': R2.item()}        
        else:
            raise ValueError
            
    def epoch_end(self, epoch, result, lrList):
        if self.output_dim == 1:  # for regression
            print("Epoch [{}], train_loss: {:.5f},"\
                  " val_loss: {:.5f}, " \
                  " R2: {:.4f},base_lr: {:.2e} ".format(
                epoch, result['train_loss'],
                result['val_loss'], result['R2'],
                lrList[0]))

        else:
           raise ValueError
           
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def test(model, val_loader):
    model.eval()
    outputs = [model.test_step(batch) for batch in val_loader]
    outputs_vector = []
    for t in outputs:
        outputs_vector.extend(t.cpu().numpy()) 
    return outputs_vector

def fit(epochs, lr, lr_decay, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = opt_func(params, lr=lr)
    # Decay LR by a factor every epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    print('Training started...')
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        
        for i,batch in enumerate(train_loader):
            loss = model.training_step(batch)
            lossTotal = loss
            train_losses.append(loss)
            lossTotal.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%1000 == 0:
                print('epoch: %s, batch: %s/%s'%(epoch,i,len(train_loader)))
        lrList = [param_group['lr'] for param_group in optimizer.param_groups]

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
      
        model.epoch_end(epoch, result,lrList)
        history.append(result)
        
        # lr decay
        exp_lr_scheduler.step()
        
    return history

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

def plot_loss(history,outFolder='',title='loss',saveFig=False):
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

def plot_R2(history,outFolder='',title='SlopeAndR2',saveFig=False):
    fig=plt.figure()
   
    val_R2 = [x['R2'] for x in history]   
 
    plt.plot(val_R2, 'y--',label = 'val_R2')
    plt.xlabel('epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.title('R2 vs. epochs')
    if saveFig:    
        fig.savefig('%s/SlopeAndR2.png'%outFolder)

def show_estimation(x,y,title='',LimRange=None):
    x=np.array(x)
    y=np.array(y)
    fig, ax = plt.subplots(1, 1,figsize = (6,5))
    R2 = []
    plt.scatter(x,y,color = 'b')

    if len(y) > 1:
        para = np.polyfit(x, y, 1)
        t=[0,1]
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
        ax.plot([np.min(x),np.max(x)], [np.min(y),np.max(y)], 'k', label='1:1 line')
    ax.legend(loc=1,edgecolor = 'w',facecolor='w', framealpha=1, ncol = 2)
    plt.xlabel('Observed', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.title(title)
    plt.legend(loc = 4)
    return fig

def obsLabel(val_dl):
    labelList = []
    for _, labels in val_dl:
        labelList.extend(labels.cpu().numpy())
    return labelList   

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
     
class loadSyntheticData():
    def __init__(self,mode, yscale,ensembleN=500):
        # generate synthetic data
   
        gen = util.genSyntheticData_fruitGrowth(note='en%s_%s_for_growthNet'%(ensembleN,mode))
        print('Loading synthetic data...')
        self.dataPair_X = [t[:2] for t in gen.dataPair_X] # 5 elements list: y0, delta_GDD, max_y, symmetry_central, growth rate
        self.dataPair_y = [t[0]*yscale for t in gen.dataPair_y]
        self.paraPair = [t[2:] for t in gen.dataPair_X]
        
        print('synthetic data loaded')
        
if __name__ == '__main__':
    #
    num_epochs = 30
    lr = 0.001
    lr_decay = 0.9
    batch_size = 512
    yscale = 1/50
    saveResult = True

    inputMode = 'GDD_Para' # 
    for mode in ['diameter','length']:
  
        now = datetime.now().strftime('%y%m%d-%H%M%S')
    
        case = '%s_growthNet_epoch%s_batch%s_%s_%s'%(mode,num_epochs,batch_size,inputMode,now)
        outPath = 'Result/%s'%case
        
        # create dataset    
        syn = loadSyntheticData(mode=mode,yscale=yscale)
        
        # test set
        X_train, X_test, y_train, y_test, para_train, para_test = train_test_split_no_shuffle(syn.dataPair_X, syn.dataPair_y, syn.paraPair, test_ratio=0.1) 
            
        train_ds = MyDataset(X=X_train, y=y_train, para=para_train, inputMode=inputMode)
        test_ds = MyDataset(X=X_test, y=y_test, para=para_test,inputMode=inputMode, test=True)
        train_dl = DataLoader(train_ds,batch_size=batch_size, shuffle=True,drop_last=True)
        test_dl = DataLoader(test_ds,batch_size=batch_size, shuffle=False)
        train_dl = DeviceDataLoader(train_dl, device)
        test_dl = DeviceDataLoader(test_dl, device)
        
        # create NN
        input_dim=2+3
        model = NN(input_dim=input_dim, output_dim=1)
        model.to(device)
        
        # train
        history = fit(num_epochs, lr, lr_decay, model, train_dl, test_dl, opt_func=torch.optim.Adam)
        if saveResult:    
            mkdir(outPath)
        plot_loss(history,outFolder=outPath,saveFig=saveResult)
        plot_R2(history,outFolder=outPath,saveFig=saveResult)
        
        # test
        pre = test(model=model, val_loader=test_dl)
        obs= obsLabel(test_dl)
        fig=show_estimation(obs,pre, LimRange=[0,1])
        if saveResult: 
            torch.save(model.state_dict(), '%s/model_state_dict.pth'%(outPath))
            torch.save(model, '%s/model.pth'%(outPath))
            fig.savefig('%s/scatterPlot.png'%(outPath))
            

