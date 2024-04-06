
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:31:38 2023

@author: yang8460

Train MLP for curve parameter learning
Input: sparse observation sequence with length from 1 to 8. The sequence will be formalized to a 1*70 vector
Output: three curve parameters
"""

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
import KGCV_util as util
import random
from scipy import stats
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

class MyDataset(Dataset):
    def __init__(self, X,y,test=False, xscale=1, yscale=1):
        self.X=X
        self.y=y
        self.test=test
        self.xscale = xscale
        self.yscale = yscale
        
    def __getitem__(self, index):
        y = np.array(self.y[index])*self.yscale
        x = np.array(self.X[index])*self.xscale
        if not self.test:
            noise = np.random.randn(len(x))*x*0.01
            x+=noise     
        return x.astype(np.float32),y.astype(np.float32)
    
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

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.criterion = nn.MSELoss()
        self.input_dim=input_dim
        self.output_dim=output_dim
        
        # Fully connected layer
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_4 = nn.Linear(32, output_dim)
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
        loss = []
        
        # for each type of label
        for i in range(self.output_dim):
            loss.append(self.criterion(y[:,i], out[:,i]))
            
        return loss
    
    def validation_step(self, batch):
        X,y = batch 
        out = self(X)                   # Generate predictions
        
        loss = []
        
        # for each type of label
        for i in range(y.shape[-1]):
            loss.append(self.criterion(y[:,i], out[:,i]).detach())
        
        return {'val_loss': loss, 
                'oriOut': out, 'oriLabels':y}
      
    def test_step(self, batch):
        X,y = batch 
        out = self(X)                    # Generate predictions
        return out
    
    def cal_R2(self,y,yhat):
        return np.corrcoef(np.array(y), np.array(yhat))[0,1]**2
    
    def validation_epoch_end(self, outputs):
        
        epochOut = torch.concat([x['oriOut'] for x in outputs],dim=0)
        epochLabels = torch.concat([x['oriLabels'] for x in outputs],dim=0)
        batch_losses_base = torch.stack([torch.stack(x['val_loss']) for x in outputs])
        
        epoch_loss_base = batch_losses_base.mean(dim=0)   # Combine losses
       
        R2 = []
        for i in range(self.output_dim):
            R2.append(self.cal_R2(epochOut[:,i].detach().cpu().numpy(), epochLabels[:,i].detach().cpu().numpy()))
      
        return {'val_loss': epoch_loss_base.detach().cpu().numpy(),
                'R2': R2}        
       
            
    def epoch_end(self, epoch, result, lrList):
       
        print("Epoch [{}], train_loss: {},"\
              " val_loss: {}, " \
              " R2: {},base_lr: {:.2e} ".format(
            epoch, ['%.4f'%t for t in result['train_loss']],
            ['%.4f'%t for t in result['val_loss']], ['%.3f'%t for t in result['R2']],
            lrList[0]))

           
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
            lossTotal = torch.sum(torch.stack(loss))
            train_losses.append([t.item() for t in loss])
            lossTotal.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%100 == 0:
                print('epoch: %s, batch: %s/%s'%(epoch,i,len(train_loader)))
        lrList = [param_group['lr'] for param_group in optimizer.param_groups]

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = np.mean(np.array(train_losses),axis=0)
      
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
    loss_train_base = np.array([x['train_loss'] for x in history])
    loss_val_base = np.array([x['val_loss'] for x in history])
    colorList = ['r','y','g','b']
    for i in range(output_dim):
        plt.plot(loss_train_base[:,i], color=colorList[i],linestyle='-',label = 'Train loss%s'%(i+1))
        plt.plot(loss_val_base[:,i], color=colorList[i],linestyle='--',label = 'Test loss%s'%(i+1))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss vs. epochs')
    if saveFig:  
        fig.savefig('%s/loss.png'%outFolder)

def plot_R2(history,outFolder='',title='SlopeAndR2',saveFig=False):
    fig=plt.figure()
    colorList = ['r','y','g','b']
    val_R2 = np.array([x['R2'] for x in history]   )
    for i in range(output_dim):
        plt.plot(val_R2[:,i], color=colorList[i],linestyle='-',label = 'val_R2-%s'%(i+1))
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

def visualizeF(model,init,days,GDD_base,RAD_base):
    model.eval()
    ytList = []
    for d in range(1,days):
        Xt = torch.tensor(np.array([init,GDD_base*d,RAD_base*d])[np.newaxis,:].astype(np.float32)).to(device)
        yt = model(Xt).detach().cpu().numpy()/yscale
        ytList.append(yt)
        
    ytList = np.array(ytList)
    plt.figure()
    plt.plot(range(1,days),ytList)
    return ytList

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class loadSyntheticData():
    def __init__(self,mode,ensembleN=500):
        # generate synthetic data
        
        gen = util.genSyntheticData_ParaLearning(note='en%s_forParaLearning_%s'%(ensembleN,mode))  # v2: observation sequence from 3-7
        print('Loading synthetic data...')
        self.dataPair_X = gen.dataPair_X # 5 elements list: y0, delta_GDD, max_y, symmetry_central, growth rate
        self.dataPair_y = gen.dataPair_y
        
        print('synthetic data loaded')

def plotScatterDense(x_, y_,alpha=1, binN=200, thresh_p = 0.05,note='',title='',uplim=None,downlim=None,
                     auxText = None,legendLoc=4, cmap='Reds', removeNeg=False,
                     vmin=None,vmax=None,removeZero=False,upcoef=1.2,plotdense=True):
        fig,ax = plt.subplots()
        x_=np.array(x_)
        y_=np.array(y_)
        if len(y_) > 1:
            if removeZero:
                loc = ((x_!=0) & (y_!=0))
                x_ = x_[loc]
                y_ = y_[loc]
                
            if removeNeg:
                loc = ((x_>0) & (y_>0))
                x_ = x_[loc]
                y_ = y_[loc]
            # Calculate the point density
            if not (thresh_p is None):
                thresh = (np.max(np.abs(x_))*thresh_p)
                loc = ((x_>thresh)|(x_<-thresh)) & ((y_>thresh)|(y_<-thresh))
                x_ = x_[loc]
                y_ = y_[loc]

            x=x_
            y=y_
            tmp = stats.linregress(x, y)
            para = [tmp[0],tmp[1]]
            # para = np.polyfit(x, y, 1)   # can't converge for large dataset
            y_fit = np.polyval(para, x)  #
            # plt.plot(x, y_fit, 'r')
        
        #histogram definition
        bins = [binN, binN] # number of bins
        
        if plotdense:
            # histogram the data
            hh, locx, locy = np.histogram2d(x, y, bins=bins)
    
            # Sort the points by density, so that the densest points are plotted last
            z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
            idx = z.argsort()
            x2, y2, z2 = x[idx], y[idx], z[idx]
    
            # plt.scatter(x2, y2, c=z2, cmap=cmap, marker='.',alpha=alpha)
            plt.scatter(x2, y2, c=z2, cmap=cmap, marker='.',alpha=alpha,vmin=vmin,vmax=vmax)
        else:
            plt.scatter(x, y)
            
        if uplim==None:
            uplim = upcoef*max(np.hstack((x, y)))
        if downlim==None:
            downlim = 0.8*min(np.hstack((x, y)))
            
        figRange = uplim - downlim
        plt.plot(np.arange(downlim-1,np.ceil(uplim)+1), np.arange(downlim-1,np.ceil(uplim)+1), 'k', label='1:1 line')
        plt.xlim([downlim, uplim])
        plt.ylim([downlim, uplim])

        if not legendLoc is None:
            if legendLoc==False:
                plt.legend(edgecolor = 'w',facecolor='w',fontsize=12)          
            else:
                plt.legend(loc = legendLoc, edgecolor = 'w',facecolor='w',fontsize=12, framealpha=0)
        plt.title(title, y=0.9, fontsize=16)
        
        if len(y) > 1:
            R2 = np.corrcoef(x, y)[0, 1] ** 2
            RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
            NRMSE = ((np.sum((y - x) ** 2) / len(y)) ** 0.5)/np.mean(x)
            Bias = np.mean(y) - np.mean(x)
            NBias = np.abs((np.mean(y) - np.mean(x))/np.mean(x))
            # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
            plt.text(downlim + 0.1 * figRange, downlim + 0.83 * figRange, r'$R^2 $= ' + str(R2)[:5], fontsize=14)
            # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.76 * figRange, r'$RMSE $= ' + str(RMSE)[:5], fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Bias $= ' + str(Bias)[:5], fontsize=14)
            
            # plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$NRMSE $= ' + str(NRMSE)[:5], fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.62 * figRange, r'$Slope $= ' + str(para[0])[:5], fontsize=14)
            # plt.text(downlim + 0.1 * figRange, downlim + 0.55 * figRange, r'$Nbais $= ' + str(NBias)[:5], fontsize=14)
            # plt.text(downlim + 0.1 * figRange, downlim + 0.62 * figRange, r'$Bias $= ' + str(Bias)[:5], fontsize=14)
        if not auxText == None:
            plt.text(0.05, 0.91, auxText, transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
        plt.colorbar()
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        return fig

def testCurvePre(test_ds, index):
    x_onehot, para = test_ds[index]
    model.eval()
    pre_para = torch.squeeze(model(torch.unsqueeze(torch.tensor(x_onehot).to(device),dim=0))).detach().cpu().numpy()/yscale
    loc = np.where(x_onehot>0)[0]
    obs_y = x_onehot[loc]/xscale
    fig = plt.figure()
    t=np.linspace(0, 600)
    plt.plot(t,util.func(t, *pre_para),'r-', label='Predicted Logistic curve')
    plt.scatter(loc*resolution,obs_y)
    plt.plot(t,util.func(t, *popt_up),'k--', label='Boundaries')
    plt.plot(t,util.func(t, *popt_d),'k--')
    plt.legend(loc=4, edgecolor = 'w')
    return fig

if __name__ == '__main__':
    
    # Hyper-parameters
    num_epochs = 100
    output_dim=3
    lr = 0.001
    lr_decay = 0.99
    batch_size = 512
    xscale = 1/50
    yscale = np.array([1/50, 1/500, -50])
    saveResult = True
    note = 'multi-hot' 
    
    for mode in ['diameter','lengrh']:   
        
        # outpath
        now = datetime.now().strftime('%y%m%d-%H%M%S')
        case = '%s_epoch%s_batch%s_%s_%s_v5'%(mode,num_epochs,batch_size,note,now)
        outPath = 'Result/%s'%case
        
        # load dataset    
        syn = loadSyntheticData(mode=mode,ensembleN=1000)
        print('Preparing the dataset...')
        dataset_X = syn.dataPair_X
        dataset_y = syn.dataPair_y
        X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_ratio=0.1) 
        print('Dataset created')
        
        # reference curve check
        getT = util.getTrajectories()
        popt, popt_up, popt_d, aux, fig = getT.fitBoundary(mode = mode, per='mannul',plot=True)
        
        # create dataset and dataloader
        train_ds = MyDataset(X=X_train, y=y_train, xscale=xscale, yscale=yscale)
        test_ds = MyDataset(X=X_test, y=y_test, test=True, xscale=xscale, yscale=yscale)
        train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds,batch_size=batch_size, shuffle=False)
        train_dl = DeviceDataLoader(train_dl, device)
        test_dl = DeviceDataLoader(test_dl, device)  
        x0, y0 = train_ds[0]
        
        # create NN    
        input_dim=len(x0)
        resolution = 700/input_dim
        model = NN(input_dim=input_dim, output_dim=output_dim)   
        model.to(device)
        
        # train
        history = fit(num_epochs, lr, lr_decay, model, train_dl, test_dl, opt_func=torch.optim.Adam)
        if saveResult:    
            mkdir(outPath)
        plot_loss(history,outFolder=outPath,saveFig=saveResult)
        plot_R2(history,outFolder=outPath,saveFig=saveResult)
        
        # test
        pre = np.array(test(model=model, val_loader=test_dl))
        obs= np.array(obsLabel(test_dl))
        fig1=plotScatterDense(obs[:,0]/yscale[0],pre[:,0]/yscale[0])
        fig2=plotScatterDense(obs[:,1]/yscale[1],pre[:,1]/yscale[1])
        fig3=plotScatterDense(obs[:,2]/yscale[2],pre[:,2]/yscale[2])
        if saveResult: 
            torch.save(model.state_dict(), '%s/model_state_dict.pth'%(outPath))
            torch.save(model, '%s/model.pth'%(outPath))
            fig1.savefig('%s/scatterPlot1.png'%(outPath))
            fig2.savefig('%s/scatterPlot2.png'%(outPath))
            fig3.savefig('%s/scatterPlot3.png'%(outPath))  
        
        ## test curve
        indexList = [t for t in range(len(test_ds))]
        random.seed(0)
        for i in range(4):
            fig1 = testCurvePre(test_ds, index=random.sample(indexList,1)[0])
            if saveResult: 
                fig1.savefig('%s/testCurvePre%s.png'%(outPath,i))