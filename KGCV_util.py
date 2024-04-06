# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 22:25:39 2024

@author: yang8460
"""
from typing import Tuple, Dict, Optional
import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F
import torchvision.models as models

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import os
import glob
import json
import copy
from scipy.optimize import leastsq
import sys
import random
version = int(sys.version.split()[0].split('.')[1])
if version > 7:
    import pickle
else:
    import pickle5 as pickle

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        data = pickle.load(inp)
    return data


class ToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target
    
def GDDcal(Tmax=None, Tmin=None,Tmean = None, Tbase = 8,Tup = 30):
        
        if Tmean is None:
            Tmax[Tmax>Tup] = Tup
            Tmax[Tmax<Tbase] = Tbase
            Tmin[Tmin>Tup] = Tup
            Tmin[Tmin<Tbase] = Tbase
            Tmean = 0.5*(Tmax + Tmin)
        else:
            Tmean[Tmean>Tup] = Tup
            Tmean[Tmean<Tbase] = Tbase
        GDD = Tmean - Tbase
        GDDcum = [np.sum(GDD[:i+1]) for i in range(len(GDD))]
        
        return GDD,GDDcum

def parseSeries(dateList, series):   
    count = 0
    dList = []
    sList = []
    for d,s in zip(dateList, series):
        if count==0:
            s_s = s
            d_tmp = [d]
            s_tmp = [s]
        
        else:            
            if s>s_s:
                d_tmp.append(d)
                s_tmp.append(s)
                s_s = s
            else:
                dList.append(d_tmp)
                sList.append(s_tmp)
                d_tmp = [d]
                s_tmp = [s]
                s_s = s
            if count==(len(series)-1):
                dList.append(d_tmp)
                sList.append(s_tmp)
        count+=1
    return dList, sList

def fetchDriver(weather,pairList_d):
    GDD_pair = []
    RAD_pair = []
    for d in pairList_d:

        tmp = weather.iloc[weather[weather['Date']==d[0]].index[0]:
                            weather[weather['Date']==d[1]].index[0]]   
        GDD_pair.append(np.sum(tmp['GDD']))
        RAD_pair.append(np.sum(tmp['RAD']))
        
    return GDD_pair, RAD_pair

def fetchDriver_series(weather,pairList_d_series):
    GDD_series = []
    RAD_series = []
    RH_series = []
    for d in pairList_d_series:       
        GDD_pair = [0]
        RAD_pair = [0]
        RH_pair = [0]
        for i in range(len(d)-1):
            tmp = weather.iloc[weather[weather['Date']==d[0]].index[0]:
                                weather[weather['Date']==d[i+1]].index[0]]   
            GDD_pair.append(np.sum(tmp['GDD']))
            RAD_pair.append(np.sum(tmp['RAD']))  
            RH_pair.append(np.mean(tmp['RHmean']))  
        GDD_series.append(GDD_pair)
        RAD_series.append(RAD_pair)
        RH_series.append(RH_pair)
        
    return GDD_series, RAD_series, RH_series

def parsePairs(df,sampleIndex = [str(t) for t in range(0,5)]):
    dateList = df['Date'].tolist()
    dList_ = []
    sList_ = []
    numList_ = []
    for i in sampleIndex:
        # parse growing cycle
        series = df[i].tolist()
        dList, sList = parseSeries(dateList, series)
        
        # sampling pair   
        for d_round,s_round in zip(dList, sList):
            if len(s_round)>1:            
                dList_.append(d_round)
                sList_.append(s_round)
                numList_.append(i)
    return dList_, sList_,numList_

def genPairs(dList, sList,numList, weather, yscale=1,aug = False):
    pairList_var = []
    pairList_d = []
    startDate = []
    for d_round,s_round,i in zip(dList, sList,numList):
        # interplate the obs, augmentation the training samples
        if aug:
            auged_d = [t.astype(datetime).strftime('%m/%d/%Y') for t in np.arange(datetime.strptime(d_round[0],'%m/%d/%Y'),
                                   datetime.strptime(d_round[-1],'%m/%d/%Y')+timedelta(1), dtype='datetime64[D]')]
            auged_doy = [datetime.strptime(t,'%m/%d/%Y').timetuple().tm_yday for t in auged_d]
            d_doy = [datetime.strptime(t,'%m/%d/%Y').timetuple().tm_yday for t in d_round]
            f_tmp = interp1d(d_doy, s_round, kind = 'linear')
            auged_s = f_tmp(auged_doy)
            
            d_round = auged_d
            s_round = list(auged_s)
        count = 0    
        for d,s in zip(d_round, s_round):
            for pair_n in range(count+1,len(s_round)):
                pairList_var.append((s,s_round[pair_n]))
                pairList_d.append((d,d_round[pair_n]))
                startDate.append((d,int(i)))
            count+=1
            
    GDD_pair, RAD_pair = fetchDriver(weather,pairList_d)
    
    X = []
    y = []
    for v,GDD,RAD in zip(pairList_var, GDD_pair, RAD_pair):
        X.append([v[0], GDD, RAD])
        y.append(v[1]*yscale)
        
    return pairList_d, X, y, startDate

def genTrajectaryGDD(dList, weather):
    GDD_list = []
    for d_round in dList:
        # interplate the obs, augmentation the training samples
        count = 0
        GDDSeries = []
        for d in d_round:                 
            pair_d=(d_round[0],d)
            GDD_pair,_ = fetchDriver(weather,[pair_d])
            GDDSeries.append(GDD_pair[0])
        count+=1
        GDD_list.append(GDDSeries)
    return GDD_list

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Dataset(object):
    '''for this case, there are multiple obs in a image but only one tagged img'''

    def __init__(self, root, year):
        self.root = root     
        self.imgs = glob.glob('%s/%s*.jpg'%(root,year))
        self.labels = ['%s.json'%(t.split('.')[0]) for t in self.imgs]
            
        
    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
      
        with open(label_path,'r') as f:
            labeldata= json.load(f)
           
        sample_id = labeldata['imagePath'].split('_')[-1].split('.')[0]
        tmp = labeldata['imagePath'].split('_')[0]
        sample_date = '%02s/%02s/%04s'%(tmp[4:6],tmp[6:8],tmp[0:4])

        tmp = [t for t in labeldata['shapes'] if not '-1' in t['label']]
        if len(tmp) == 0:
            label = labeldata['shapes'][0]
        else:
            label = tmp[0]
        labels_pheno=label['label'].split(', ')[0]
        labels_diameter=np.float32(label['label'].split(', ')[1])
        labels_len=np.float32(label['label'].split(', ')[2])
               
        return labels_pheno, labels_diameter, labels_len, sample_id, sample_date 

    def __len__(self):
        return len(self.imgs)

def loadPhenoData(all_ds,validlabel):
    label_list = [all_ds[i] for i in range(len(all_ds))]
    tmp_df = pd.DataFrame(label_list,columns=['labels_pheno', 'labels_diameter', 
                                         'labels_len', 'sample_id', 'sample_date'])
    dateList = list(set(tmp_df['sample_date'].tolist()))
    dateList.sort()
    pheno_dic = {}
    for d in dateList:
        tmp = tmp_df[tmp_df['sample_date']==d]
        # to convert lists to dictionary
        tmp_dic = {tmp['sample_id'].tolist()[i]: tmp['labels_pheno'].tolist()[i] for i in range(len(tmp))}
        pheno_dic[d]=tmp_dic
    pheno_dic_ = {}
    pheno_dic_['Date'] = dateList
    for i in range(1,21):
        sample_series = []
        for d in dateList:
            tmp = pheno_dic[d].get(str(i),np.nan)
            if tmp is np.nan:
                sample_series.append(tmp)
            else:
                sample_series.append(validlabel.index(tmp))
        pheno_dic_[str(i-1)] = sample_series
    pheno_df = pd.DataFrame(pheno_dic_)
    return pheno_df

def getStart(pheno_df):
    loc = [str(t) for t in range(20)]
    tmp = np.where(np.array(pheno_df[loc])==1)
    pheno_dates = pheno_df['Date'].tolist()
    small_g_index = [(pheno_dates[d],s) for d,s in zip(tmp[0],tmp[1])]
    
    return small_g_index

def plot_GDD_size(X_sgs,y_sgs,xaxis,xlabel):
    GDD_all = []
    RAD_all = []
    y_all = []
    plt.figure()
    colorList = ['c','y','b','g','r']
    Nlevels = ['0N','50N','100N','150N']
    for X_sg, y_sg,c,N in zip(X_sgs,copy.deepcopy(y_sgs),colorList,Nlevels):
        basedSize = np.array(X_sg)[:,0]
        y_sg.extend(basedSize)
        normalized_y = np.array(y_sg)
        # x = np.array(X_sg)[:,1]
        # normalized_y = np.array(y_sg) - basedSize
        tmp = [t[2] for t in X_sg]
        tmp.extend([0.0]*len(basedSize))
        x1 = np.array(tmp) + 30
        RAD_all.extend(x1)
            
        tmp = [t[1] for t in X_sg]
        tmp.extend([0.0]*len(basedSize))
        x2 = np.array(tmp) + 100
        GDD_all.extend(x2)
        if xaxis=='GDD':
            plt.scatter(x2, normalized_y, color=c,label=N)
        else:
            plt.scatter(x1, normalized_y, color=c,label=N)
        
        y_all.extend(normalized_y)
    plt.legend(edgecolor = 'w',facecolor='w')
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel('Fruit diameter (mm)',fontsize=15)
    return np.array(GDD_all), np.array(RAD_all), np.array(y_all)

def plot_trajectary(trajectarys, flowerGDD=100,plot=False):
    if plot:
        fig = plt.figure()
    colorList = ['c','y','b','g','r']
    Nlevels = ['0N','50N','100N','150N']
    GDD_all = []
    y_all = []
    for t,c,N in zip(trajectarys,colorList,Nlevels):
        for i,tt in enumerate(t):
            date, y, GDD = tt
            x=np.array(GDD)+flowerGDD
            y=np.array(y)
            if plot:
                if i==0:
                    plt.plot(x, y, color=c,label=N,marker='.',markersize=8, alpha=0.2)
                else:
                    plt.plot(x, y, color=c,marker='.',markersize=8, alpha=0.4)
            GDD_all.extend(x)
            y_all.extend(y)
    # plt.legend(edgecolor = 'w',facecolor='w')
    if plot:
        return np.array(GDD_all), np.array(y_all),fig
    else:
        return np.array(GDD_all), np.array(y_all),None

def plot_trajectary_pheno(trajectarys, flowerGDD=100, plot=True,validlabel=''):
    if plot:
        plt.figure()
    colorList = ['c','y','b','g','r','m','orange']
   
    GDD_all = []
    y_all = []
    appearedPheno = []
    for i,tt in enumerate(trajectarys):
        date, y, GDD, _,_, pheno,_ = tt
        x=np.array(GDD)+flowerGDD
        y=np.array(y)
        if plot:
            # plt.plot(x, y, color='grey', marker='.', markersize=6, alpha=0.2)
            plt.plot(x, y, color='grey', alpha=0.2)
            nonanLoc = ~np.isnan(pheno)
            x_,y_,p_ = x[nonanLoc],y[nonanLoc],pheno[nonanLoc]
            for j,k,l in zip(x_,y_,p_):
                l=int(l)
                if l not in appearedPheno:
                    plt.scatter(j, k, color=colorList[l],label=validlabel[l],marker='.',s=20, alpha=0.5)
                else:
                    plt.scatter(j, k, color=colorList[l],marker='.',s=30, alpha=0.5)
                appearedPheno.append(l)              
        GDD_all.extend(x)
        y_all.extend(y)
    if plot:
        plt.legend(edgecolor = 'w',facecolor='w')
        
    return np.array(GDD_all), np.array(y_all)

def func(x,y0,t0,r,s=-4):
    # s=-4
    fx = (y0+s)/(1 + np.exp(r*(x-t0)))-s
    # fx = (y0)/(1 + np.exp(r*(x-t0)))
    return fx

def func_reverse(y,y0,t0,r):
    # fx = (y0+s)/(1 + np.exp(r*(x-t0)))-s
    s=-4
    x = np.log((y0+s)/(y+s) - 1)/r + t0
    return x

def loss_fit(p,x,y,xo=0,yo=6):
    mse = (y-func_fit(x,*p))**2
    point = (yo - func_fit(xo,*p))**2
    return mse + point

def func_fit(x,y0,t0,r):
    return func(x,y0,t0,r)

# def loss_fit(p,x,y):
#     return (y-func_fit(x,*p))**2
def selectParaCombination(p,xo=0,yo=4,thresh = 0.5):
    point = np.abs(yo - func(xo,*p))
    if point < thresh:
        return True
    else:
        return False

def selectParaCombinationRange(p,xo=0,yo_up=4,yo_d=3):
    yp = func(xo,*p)
    if yp>=yo_d and yp<=yo_up:
        return True
    else:
        return False
    
def separatePoints(x,y,p):
    y_ = func(x, *p)
    x_percentile_up = []
    y_percentile_up = []
    x_percentile_d = []
    y_percentile_d = []
    for xx,yy,yy_ in zip(x,y,y_):
        if yy>=yy_:
            x_percentile_up.append(xx)
            y_percentile_up.append(yy)
        else:
            x_percentile_d.append(xx)
            y_percentile_d.append(yy)
    return np.array(x_percentile_up), np.array(y_percentile_up), np.array(x_percentile_d), np.array(y_percentile_d)

def supplementZero(x,v=0,n=20):
    x = list(x)
    x.extend([v]*n)
    return np.array(x)

class genSyntheticData_fruitGrowth():
    def __init__(self,paraList=None,note='', force = False, x_num=200):
        self.paraList=paraList
        self.note=note
        self.x_num = x_num
        # self.genData()
        if (os.path.exists('datasets/synthetic_Xpair_%s.pkl'%self.note)) & (not force):
            self.dataPair_X = load_object('datasets/synthetic_Xpair_%s.pkl'%self.note)
            self.dataPair_y = load_object('datasets/synthetic_ypair_%s.pkl'%self.note)
        else:
            self.genData()
            
    def genData(self):
        x0_list=np.linspace(0, 700, num = self.x_num)
        
        self.dataPair_X = []
        self.dataPair_y = []
        for n, para in enumerate(self.paraList):
            
            # disturb elements of [2:-1]
            resolution = 700/self.x_num
            x0_list_disturb = x0_list.copy()
            for i,t in enumerate(x0_list):
                if i>0:
                    x0_list_disturb[i]=t+random.uniform(-resolution*0.5,resolution*0.5)

            y0_list = func(x0_list_disturb, *para)
            # plt.plot(t,func(t, *para),'r-')
            # gen pair for one curve
            pair_X = []
            pair_y = []
            for i in range(len(x0_list_disturb)):
                restIndex = np.arange(i+1,len(x0_list_disturb))
                for t in restIndex:
                    pair_X.append([y0_list[i],x0_list_disturb[t]-x0_list_disturb[i],*para])
                    pair_y.append([y0_list[t]])
            self.dataPair_X.extend(pair_X)
            self.dataPair_y.extend(pair_y)
            
            print('Finished %.2f percent'%(100*(n+1)/len(self.paraList)))
        # saving
        save_object(self.dataPair_X,'datasets/synthetic_Xpair_%s.pkl'%self.note)
        save_object(self.dataPair_y,'datasets/synthetic_ypair_%s.pkl'%self.note)
        
class genSyntheticData_ParaLearning():
    def __init__(self,paraList=None,resolution=10,intervalRange = [3,15], obsLimit = [1,8], samplingPerCurve = 200, note='', force = False, exact=False):
        self.paraList=paraList
        self.resolution = resolution
        self.obsLimit=obsLimit
        self.intervalRange = intervalRange
        self.samplingPerCurve=samplingPerCurve
        self.startLimit = [0,200]
        self.exact = exact
        # self.genData()
        self.note=note
        ensembleN = int(note.split('en')[1].split('_')[0])
        if (os.path.exists('datasets/syntheticData/synthetic_Xpair_%s.pkl'%self.note)) & (not force):
            self.dataPair_X = load_object('datasets/syntheticData/synthetic_Xpair_%s.pkl'%self.note)
            self.dataPair_y = load_object('datasets/syntheticData/synthetic_ypair_%s.pkl'%self.note)
        else:
            print('synthetic data not exist, now generating...')
            self.paraList = get_paraList(ensembleN = ensembleN, mode = 'diameter')
            self.genData()
            
    def genData(self):
        np.random.seed(0)
        x0_list = np.arange(0,700,self.resolution).astype(np.float32)
        self.dataPair_X = []
        self.dataPair_y = []
        for n, para in enumerate(self.paraList):
            para = para.astype(np.float32)
            y0_list = func(x0_list, *para).astype(np.float32)
            # plt.plot(t,func(t, *para),'r-')
            # gen pair for one curve
            pair_X = []
            pair_y = []
            x_s_curve = []
            for i in range(self.obsLimit[0],self.obsLimit[1]+1):
 
                # sampling the start point
                count = 0
                while True:
                    x_s = [np.random.uniform(self.startLimit[0],self.startLimit[1])]
                    for j in range(i-1):
                        delta = np.random.uniform(self.intervalRange[0],self.intervalRange[1])*self.resolution
                        x_s.append(x_s[-1] + delta)
                        
                    x_index = np.round(np.array(x_s)/self.resolution).astype(np.int64)
                    if self.exact:
                        x_s = x_index.astype(np.float32)*self.resolution
                    if np.max(x_s) <= 600:
                        count+=1
                        y_s = func(x_s, *para).astype(np.float32)
                        tmp = np.zeros(len(x0_list))
                        tmp[x_index] = y_s
                        pair_X.append(tmp)
                        pair_y.append(para)
                        x_s_curve.append(x_s)
                    if count >= self.samplingPerCurve:
                        break
                # tmp = random.sample(list(x0_list),3)
                
            self.dataPair_X.extend(pair_X)
            self.dataPair_y.extend(pair_y)
            
            if n%20 == 0:
                print('Finished %.2f percent'%(100*(n+1)/len(self.paraList)))
        # saving
        save_object(self.dataPair_X,'datasets/syntheticData/synthetic_Xpair_%s.pkl'%self.note)
        save_object(self.dataPair_y,'datasets/syntheticData/synthetic_ypair_%s.pkl'%self.note)

def parseSeriesPheno(dateList, series, seriesPheno):   
    count = 0
    dList = []
    sList = []
    pList = []
    for d,s,p in zip(dateList, series, seriesPheno):
        if count==0:
            s_s = s
            d_tmp = [d]
            s_tmp = [s]
            p_tmp = [p]
        else:            
            if s>s_s:
                d_tmp.append(d)
                s_tmp.append(s)
                p_tmp.append(p)
                s_s = s
            else:
                dList.append(d_tmp)
                sList.append(s_tmp)
                pList.append(p_tmp)
                d_tmp = [d]
                s_tmp = [s]
                p_tmp = [p]
                s_s = s
            if count==(len(series)-1):
                dList.append(d_tmp)
                sList.append(s_tmp)
                pList.append(p_tmp)
        count+=1
    return dList, sList, pList

def parsePairs2023(df,df_pheno,sampleIndex = [str(t) for t in range(0,5)]):
    df = copy.deepcopy(df)
    df_pheno = copy.deepcopy(df_pheno)
    df.fillna(0, inplace=True)
    dateList = df['Date'].tolist()
    dList_ = []
    sList_ = []
    pList_ = []
    numList_ = []
    for i in sampleIndex:
        # parse growing cycle
        series = df[i].tolist()
        series_pheno = df_pheno[i].tolist()
        dList, sList, pList = parseSeriesPheno(dateList, series, series_pheno)
        
        # sampling pair   
        for d_round,s_round,p_round in zip(dList, sList, pList):
            if len(s_round)>1:            
                dList_.append(d_round)
                sList_.append(s_round)
                pList_.append(p_round)
                numList_.append(i)
    return dList_, sList_, pList_, numList_

class getTrajectories():
    def __init__(self):
        #    
        self.Nlevels = ['0N','50N','100N','150N']
        self.validlabel = ['flower','small g','green','white','turning red','red','overripe']  
        
        # load data
        self.weather2022 = pd.read_csv('datasets/measurements/weather_daily_2022.csv')
        self.diameter2022 = pd.read_csv('datasets/measurements/data_taggedFruit_diameter_2022.csv')
        self.length2022 = pd.read_csv('datasets/measurements/data_taggedFruit_length_2022.csv')
        
        self.weather2023 = pd.read_csv('datasets/measurements/weather_daily_2023.csv')
        self.diameter2023 = pd.read_csv('datasets/measurements/data_taggedFruit_diameter_2023.csv')
        self.length2023 = pd.read_csv('datasets/measurements/data_taggedFruit_length_2023.csv')
        
        # phenology label to dataframe
        self.pheno_df2022 = loadPhenoData(Dataset(root = 'datasets/strawberry_img_tagged',year=2022),self.validlabel)
        self.pheno_df2023 = loadPhenoData(Dataset(root = 'datasets/strawberry_img_tagged',year=2023),self.validlabel)
        self.pheno_df = pd.concat([self.pheno_df2022,self.pheno_df2023], ignore_index=True)
        
        # calculate GDD
        self.GDD2022,_ = GDDcal(Tmean = self.weather2022['Tmean'].values, Tbase = 3,Tup = 40)
        self.weather2022['GDD'] = self.GDD2022
        self.GDD2023,_ = GDDcal(Tmean = self.weather2023['Tmean'].values, Tbase = 3,Tup = 40)
        self.weather2023['GDD'] = self.GDD2023
        self.weather = pd.concat([self.weather2022,self.weather2023], ignore_index=True)
        
    def retrieveTraj(self, mode = 'diameter', yscale=1):
        # retrieve trajectories
        self.trajectarys = []
        self.smallG_GDD_list = []
        for Nlevel in self.Nlevels:
            # generate the sample pairs (l_t2, l_t1, delta_GDD, delta_RAD)
            if Nlevel == '0N':
                sampleIndex = [str(t) for t in range(0,5)]
            elif Nlevel == '50N':
                sampleIndex = [str(t) for t in range(5,10)]
            elif Nlevel == '100N':
                sampleIndex = [str(t) for t in range(10,15)]
            elif Nlevel == '150N':
                sampleIndex = [str(t) for t in range(15,20)]
            elif Nlevel == 'all':
                sampleIndex = [str(t) for t in range(0,20)]
            
            if mode == 'diameter':
                df2022=self.diameter2022
                df2023=self.diameter2023
                size_thresh_up = 9.5
                size_thresh_low = 6.5
                self.anchor=[0,6]
            elif mode == 'length':
                df2022 = self.length2022
                df2023 = self.length2023
                size_thresh_up = 12.5
                size_thresh_low = 6.5
                self.anchor=[0,6]
                
            # 2022
            dList2022, sList2022,numList2022 = parsePairs(df2022,sampleIndex = sampleIndex)
            pairList_d2022, X2022, y2022,startDate2022 = genPairs(dList2022, sList2022,numList2022, weather=self.weather2022,
                                          yscale=yscale,aug=False)
            GDD_list2022 = genTrajectaryGDD(dList2022, self.weather2022)
            
            # 2023
            dList2023, sList2023, numList2023 = parsePairs(df2023,sampleIndex = sampleIndex)
            pairList_d2023, X2023, y2023,startDate2023 = genPairs(dList2023, sList2023,numList2023, weather=self.weather2023,
                                          yscale=yscale,aug=False)
            GDD_list2023 = genTrajectaryGDD(dList2023, self.weather2023)
            
            # determin GDD from flower to small g
            dList2023_f, sList2023_f, pList2023_f, numList2023_f = parsePairs2023(df2023,self.pheno_df2023,sampleIndex = sampleIndex)    
            GDD_list2023_f = genTrajectaryGDD(dList2023_f, self.weather2023)
            for j,k in zip(GDD_list2023_f, pList2023_f):
                if k[0] == 0:
                    if 1 in k:
                        self.smallG_GDD_list.append(j[k.index(1)])
            
            # filter trajectaries      
            trajectary2022 = []
            for d,s,g,n in zip(dList2022, sList2022, GDD_list2022,numList2022):
                if (s[0] < size_thresh_up) & (s[0] > size_thresh_low):
                    trajectary2022.append([d,s,g])
    
            trajectary2023 = []
            for d,s,g,n in zip(dList2023, sList2023, GDD_list2023,numList2023):
                if (s[0] < size_thresh_up) & (s[0] > size_thresh_low):
                    trajectary2023.append([d,s,g])
    
            trajectary = trajectary2022 + trajectary2023        
            self.trajectarys.append(trajectary)
    
    def retrieveHighQualityTraj(self, seriesThresh = 3,size_thresh_up=9.5,size_thresh_low = 6.5):
    
        # extract the high quality trajectories
        yscale = 1
        mode = 'diameter' # 
      
        # generate the sample pairs (l_t2, l_t1, delta_GDD, delta_RAD)
        sampleIndex = [str(t) for t in range(0,20)]
        
        if mode == 'diameter':
            df2022=self.diameter2022
            df2023=self.diameter2023
            # size_thresh_up = 9.5
            # size_thresh_low = 6.5
        elif mode == 'length':
            df2022 = self.length2022
            df2023 = self.length2023
            # size_thresh_up = 9.5
            # size_thresh_low = 5
        valid_traj_list=[]
        for df,weather,pheno_df in zip([df2022,df2023],[self.weather2022,self.weather2023],[self.pheno_df2022,self.pheno_df2023]):
            pairList_d, pairList_id, pairList_var, pairList_pheno, GDD_pair, RAD_pair, RH_pair = self.genSeries(df=df,weather=weather,
                                          sampleIndex = sampleIndex, yscale=yscale,seriesThresh = seriesThresh,pheno_df=pheno_df,
                                          size_thresh_up=size_thresh_up,size_thresh_low=size_thresh_low)
            
            # statistics of pheno duration
            duration = []
            for startPheno in range(1,5):
                tmp = []
                for p, g in zip(pairList_pheno, GDD_pair):
                    p = list(p)
                    if (startPheno in p) & ((startPheno+1) in p):
                        tmp.append(g[p.index(startPheno+1)] - g[p.index(startPheno)])
                duration.append(tmp)        
            
            # required GDD 
            requiredGDD = [np.mean(t) for t in duration]
            print(requiredGDD)
        
            trajectarys=[[i,j,k,a,b,x,y] for i,j,k,a,b,x,y in zip(pairList_d,pairList_var,GDD_pair,RAD_pair,RH_pair,pairList_pheno,pairList_id)]
            GDD,y = plot_trajectary_pheno(trajectarys, flowerGDD=100,validlabel=self.validlabel)
           
            # filter the pair, requirement: monotone increase
            valid_traj = []
            for t in trajectarys:
                t_pheno = t[-2]
                tmp = t_pheno[~np.isnan(t_pheno)]
                tmp_2 = list(tmp.copy())
                tmp_2.sort()
                if list(tmp) == tmp_2:
                    valid_traj.append(t)
            valid_traj_list.append(valid_traj)
        
        # cal lifespan
        complete_traj_list = []
        meanlifespan = []
        for trajectorys in valid_traj_list:
            complete_traj = [t for t in trajectorys if (len(t[-2])>4) and (t[-2][-1]>=5) and (t[-2][0]==1)]
            complete_traj_list.append(complete_traj)
            traj_lifespan = [t for t in complete_traj if t[5][-1]==5]
            meanlifespan.append(np.mean([t[2][-1] for t in traj_lifespan]) +80)

        return valid_traj_list, complete_traj_list, meanlifespan

    def genSeries(self, df,weather, sampleIndex = [str(t) for t in range(0,5)],
              yscale=1,seriesThresh = 5,pheno_df='',size_thresh_up=None,size_thresh_low=None):
        dateList = df['Date'].tolist()
        
        pairList_var = []
        pairList_d = []
        pairList_pheno = []
        pairList_id = []
    
        for i in sampleIndex:
            # parse growing cycle
            series = df[i].tolist()
            series_pheno = pheno_df[i].tolist()
            dates_pheno = pheno_df['Date'].tolist()
            dList, sList = parseSeries(dateList, series)
            
            # sampling pair   
            for d_round,s_round in zip(dList, sList):
                if len(s_round)>=seriesThresh:
                    if (s_round[0] < size_thresh_up) & (s_round[0] > size_thresh_low):                        
                        pairList_var.append(s_round)
                        pairList_d.append(d_round)
                        
                        # gen pheno for pair
                        tmp = np.ones(len(s_round))*np.nan
                        for j,d in enumerate(d_round):
                            if d in dates_pheno:
                                tmp[j]=series_pheno[dates_pheno.index(d)]
                        pairList_pheno.append(tmp)
                        pairList_id.append(i)
        GDD_pair, RAD_pair, RH_pair = fetchDriver_series(weather,pairList_d)
         
        return pairList_d,pairList_id, pairList_var, pairList_pheno, GDD_pair, RAD_pair, RH_pair

    def fitBoundary(self, mode = 'diameter',saveResult = False,per='mannul', plot=False):
        self.retrieveTraj(mode = mode, yscale=1)
        
        # plot size-GDD curve
        t=np.linspace(0, 600)  
        GDD,y,fig = plot_trajectary(self.trajectarys, flowerGDD=np.mean(self.smallG_GDD_list),plot=plot)
        if plot:
            plt.xlabel('GDD after flowering (℃·day)')
            plt.ylabel('%s mm'%mode)
        print('mean small g GDD is %s'%np.mean(self.smallG_GDD_list))
    
        # fit the curve
        x = GDD
        # intial curve parameters
        p0 = [19, 200, -0.02]
        # fitting    
        popt=leastsq(loss_fit,p0,args=(x,y,self.anchor[0],self.anchor[1]))[0]
        print(*popt)
        if plot:
            plt.plot(t,func(t, *popt),'r-', label='Fitted Logistic curve')
                
        if per == 'mannul': 
            if mode == 'diameter':
                popt_up = [32,140,-0.017]
                popt_d = [21,410,-0.009]
            else:
                popt_up = [40,170,-0.014]
                popt_d = [22,400,-0.007]
            if plot:
                plt.plot(t,func(t, *popt_up),'k--', label='Boundaries')
                plt.plot(t,func(t, *popt_d),'k--')
            return popt, popt_up, popt_d, [t, x, y, self.weather, self.trajectarys, self.smallG_GDD_list],fig
        
        # 75% percentile curve
        x_percentile_up, y_percentile_up, x_percentile_d, y_percentile_d = separatePoints(x,y,popt)
        popt_up=leastsq(loss_fit,popt,args=(x_percentile_up, y_percentile_up,self.anchor[0],self.anchor[1]))[0]
        popt_d=leastsq(loss_fit,popt,args=(x_percentile_d, y_percentile_d,self.anchor[0],self.anchor[1]))[0]
        
        # 87.5% percentile curve
        x_percentile_up_875, y_percentile_up_875,_,_ = separatePoints(x_percentile_up, y_percentile_up,popt_up)
        popt_up_875=leastsq(loss_fit,popt,args=(x_percentile_up_875, y_percentile_up_875,self.anchor[0],self.anchor[1]))[0]
        
        _,_,x_percentile_d_875, y_percentile_d_875 = separatePoints(x_percentile_d, y_percentile_d,popt_d)
        popt_d_875=leastsq(loss_fit,popt,args=(x_percentile_d_875, y_percentile_d_875,self.anchor[0],self.anchor[1]))[0]
        
        # 93.75% percentile curve
        x_percentile_up_94, y_percentile_up_94,_,_ = separatePoints(x_percentile_up_875, y_percentile_up_875,popt_up_875)
        popt_up_94=leastsq(loss_fit,popt,args=(x_percentile_up_94, y_percentile_up_94,self.anchor[0],self.anchor[1]))[0]
        
        _,_,x_percentile_d_94, y_percentile_d_94 = separatePoints(x_percentile_d_875, y_percentile_d_875,popt_d_875)
        popt_d_94=leastsq(loss_fit,popt,args=(x_percentile_d_94, y_percentile_d_94,self.anchor[0],self.anchor[1]))[0]
        
        if per == '875':
            if plot:
                plt.plot(t,func(t, *popt_up_875),'k--', label='Boundaries')
                plt.plot(t,func(t, *popt_d_875),'k--')
            return popt, popt_up_875, popt_d_875, [t, x, y, self.weather, self.trajectarys, self.smallG_GDD_list],fig
        elif per == '94':
            if plot:
                plt.plot(t,func(t, *popt_up_94),'k--', label='Boundaries')
                plt.plot(t,func(t, *popt_d_94),'k--')
            return popt, popt_up_94, popt_d_94, [t, x, y, self.weather, self.trajectarys, self.smallG_GDD_list],fig
        elif per == 'mannul': 
            if mode == 'diameter':
                popt_up = [32,140,-0.017]
                popt_d = [21,410,-0.009]
            else:
                popt_up = [40,170,-0.014]
                popt_d = [22,400,-0.007]
            if plot:
                plt.plot(t,func(t, *popt_up),'k--', label='Boundaries')
                plt.plot(t,func(t, *popt_d),'k--')
            return popt, popt_up, popt_d, [t, x, y, self.weather, self.trajectarys, self.smallG_GDD_list],fig
    
def getFlower(pheno_df):
    loc = [str(t) for t in range(20)]
    tmp = np.where(np.array(pheno_df[loc])==0)
    pheno_dates = pheno_df['Date'].tolist()
    flower_index = [(pheno_dates[d],s) for d,s in zip(tmp[0],tmp[1])]
    
    return flower_index

def samplingPara(popt,popt_up,popt_d,t=np.linspace(0, 600),ensembleN = 100 ,mode='diameter'):    
    xo=0
    yo_up = func(xo, *popt_up)        
    yo_d = func(xo, *popt_d)
    xe=600
    ye_up = func(xe, *popt_up)        
    ye_d = func(xe, *popt_d)  
    
    if mode == 'diameter':    
        spanRatio = 0.5
    elif mode == 'length':
        spanRatio = 0.5
    
    np.random.seed(1)
    perturbRange = [[np.min([u,d]), np.max([u,d])] for u,d in zip(popt_up,popt_d)]
    # perturbRange[1][0] = 140
    # perturbRange[1][1] = 260
    paraList = []
    fig = plt.figure()
    
    # count = 0
    countValid = 0
    while True:
        para = []
        for n,p_r in enumerate(perturbRange):
            # sampling the para
            span = np.abs(p_r[0]-p_r[1])
            if n==1:
                para.append(np.random.uniform(p_r[0]-span*0.1,p_r[1]+span*0.1))
            else:
                para.append(np.random.uniform(p_r[0]-span*spanRatio,p_r[1]+span*spanRatio))
        if selectParaCombinationRange(para,xe,ye_up,ye_d) and selectParaCombinationRange(p=para,xo=xo,yo_up=yo_up,yo_d = yo_d) :   
            paraList.append(np.array(para))
            if countValid == 0:
                plt.plot(t,func(t, *para),color='grey',alpha=0.1,label='Possible growth curves')
            else:
                plt.plot(t,func(t, *para),color='grey',alpha=0.1)
            countValid+=1
        # count+=1
        if countValid>=ensembleN:
            break
    plt.plot(t,func(t, *popt),'r-', label='Fitted Logistic curve')
    plt.plot(t,func(t, *popt_up),'k--', label='Boundaries')
    plt.plot(t,func(t, *popt_d),'k--')
    return paraList,fig

class NN_para(nn.Module):
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
            
class NN_growth(nn.Module):
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

class ImageClassificationModel(nn.Module):
    def __init__(self,model='resnet18',criterion = nn.CrossEntropyLoss(), outNum = 1):
        super().__init__()
        # Use a pretrained model
        self.lastfc=0
        if model=='inception_v3':
            self.network = models.inception_v3(pretrained=True,aux_logits=False)
        elif model=='resnet18':
            self.network = models.resnet18(pretrained=True)
        elif model=='resnet50':
            self.network = models.resnet50(pretrained=True)
        elif model=='densenet121':
            self.network = models.densenet121(pretrained=True)
            self.lastfc=1
        else:
            raise ValueError('Model not available.')
        # Replace last layer
        if self.lastfc==0:
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, outNum)
        elif self.lastfc==1:
            num_ftrs = self.network.classifier.in_features
            self.network.classifier = nn.Linear(num_ftrs, outNum)
        else:
            raise ValueError('Situation not available.')
        self.criterion = criterion
        self.outNum = outNum
        
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)               # Generate predictions
        y = torch.squeeze(labels[0])
        yhat = out[:,0]
        mask = (y.detach() >= 0).float()
        if torch.sum(mask)==0:
            loss_dia = 0
        else:
            loss_dia = torch.sum(((yhat-y)*mask)**2) / torch.sum(mask)#self.criterion(y, yhat)
        
        y = torch.squeeze(labels[1])
        yhat = out[:,1]
        mask = (y.detach() >= 0).float()
        if torch.sum(mask)==0:
            loss_len = 0
        else:
            loss_len = torch.sum(((yhat-y)*mask)**2) / torch.sum(mask)#self.criterion(torch.squeeze(labels[1]), out[:,1])
        
        y = torch.squeeze(labels[2])
        yhat = out[:,2]
        mask = (y.detach() >= 0).float()
        if torch.sum(mask)==0:
            loss_pheno = 0
        else:
            loss_pheno = torch.sum(((yhat-y)*mask)**2) / torch.sum(mask)#self.criterion(torch.squeeze(labels[1]), out[:,1])


        loss = loss_dia + loss_len + loss_pheno
        return loss
    
    def forward(self, xb):
        if self.outNum == 1:  # for regression
            return torch.squeeze(torch.sigmoid(self.network(xb)))
        else:
            return self.network(xb)
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        y = torch.squeeze(labels[0])
        yhat = out[:,0]
        mask = (y.detach() >= 0).float()
        if torch.sum(mask)==0:
            loss_dia = 0
        else:
            loss_dia = torch.sum(((yhat-y)*mask)**2) / torch.sum(mask)#self.criterion(y, yhat)
        
        y = torch.squeeze(labels[1])
        yhat = out[:,1]
        mask = (y.detach() >= 0).float()
        if torch.sum(mask)==0:
            loss_len = 0
        else:
            loss_len = torch.sum(((yhat-y)*mask)**2) / torch.sum(mask)#self.criterion(torch.squeeze(labels[1]), out[:,1])
        
        y = torch.squeeze(labels[2])
        yhat = out[:,2]
        mask = (y.detach() >= 0).float()
        if torch.sum(mask)==0:
            loss_pheno = 0
        else:
            loss_pheno = torch.sum(((yhat-y)*mask)**2) / torch.sum(mask)#self.criterion(torch.squeeze(labels[1]), out[:,1])

        loss = loss_dia + loss_len + loss_pheno
        
      
        return {'val_loss': loss.detach(), 
                'oriOut': out, 'oriLabels':labels}       

    def test_step(self, batch):
        images, _ = batch
        img = torch.concat(images,dim=0)
        out = self(img)                    # Generate predictions
        return out
      
    def validation_epoch_end(self, outputs):

        if self.outNum == 1:  # for regression
            R2, epoch_loss_base = self.out_stat(outputs)
            return {'val_loss': epoch_loss_base.item(),
                    'R2': R2.item()}        
        elif self.outNum >= 2:  # for regression
            R2_dia, epoch_loss_base = self.out_stat(outputs,index=0)
            R2_len, _ = self.out_stat(outputs,index=1)
            R2_pheno, _ = self.out_stat(outputs,index=2)
            return {'val_loss': epoch_loss_base.item(),
                    'R2_dia': R2_dia.item(),                    
                    'R2_len': R2_len.item(),
                    'R2_pheno': R2_pheno.item()}   
        else:
            raise ValueError
       
    def out_stat(self,outputs,index = None):
        if self.outNum == 1:
            epochOut = [x['oriOut'] for x in outputs]
            epochLabels = [x['oriLabels'] for x in outputs]
        else:
            epochOut = [x['oriOut'][:,index] for x in outputs]
            epochLabels = [torch.squeeze(x['oriLabels'][index]) for x in outputs]
        outVec = []
        labelsVec = []
        batch_losses_base = [x['val_loss'] for x in outputs]
        epoch_loss_base = torch.stack(batch_losses_base).mean()   # Combine losses
       
        for t,k in zip(epochOut,epochLabels):
            outVec.extend(t.cpu().numpy())
            labelsVec.extend(k.cpu().numpy())
        obs,pre = removeNaN(obs=labelsVec, pre=outVec)
        R2 = np.corrcoef(obs, pre)[0,1]**2
        R2 = torch.from_numpy(np.array(R2))
        return R2, epoch_loss_base
    
    def epoch_end(self, epoch, result, lrList):
        if self.outNum == 1:  # for regression
            print("Epoch [{}], train_loss: {:.4f},"\
                  " val_loss: {:.4f}, " \
                  " R2: {:.4f}, fc_lr: {:.6f}, base_lr: {:.2e} ".format(
                epoch, result['train_loss'],
                result['val_loss'], result['R2'],
                lrList[0],lrList[1]))
                
        if self.outNum >= 2:  # for regression
            print("Epoch [{}], train_loss: {:.4f},"\
                  " val_loss: {:.4f}, " \
                  " R2_dia: {:.4f}, R2_len: {:.4f}, R2_peno: {:.4f}, fc_lr: {:.6f}, base_lr: {:.2e} ".format(
                epoch, result['train_loss'],
                result['val_loss'], result['R2_dia'],result['R2_len'],result['R2_pheno'],
                lrList[0],lrList[1]))
        else:
           raise ValueError   

def removeNaN(obs,pre):
    
    x = np.array(obs)
    y = np.array(pre)
    x[x<0] = np.nan
    
    Loc = (1 - (np.isnan(x) | np.isnan(y)))
    x_ = x[Loc==1]
    y_ = y[Loc==1]
    return x_,y_

def calHavestTime_continuous(mean_dailyGDD,currentGDD,pheno,plot=False,meanlifespan=None):
    # pheno
    validlabel = ['flower','small g','green','white','turning red','red','overripe']  # 0-6.9
    all_ds = Dataset(root = 'datasets/strawberry_img_tagged',year=2023)
    
    pheno_df = loadPhenoData(all_ds,validlabel)
    percentPheno = barPlotCluster2023(pheno_df,obslabel=validlabel,plot=plot)
    
    # mid point of each phenological stage
    progress = []
    pheno_node = [0]
    for i,t in enumerate(percentPheno):
        progress.append(np.sum(percentPheno[:i]))
        pheno_node.append(i+1)
    progress.append(1)
    f_progress = interp1d(pheno_node, y=progress, bounds_error=False, fill_value='extrapolate')
    
    # append remain GDD
    current_progress = f_progress(pheno)
    if meanlifespan is None:
        remainedGDD = currentGDD/current_progress*(1-current_progress)
    else:
        remainedGDD = meanlifespan*(1-current_progress)
      
    remainedDay = remainedGDD/mean_dailyGDD
   
    return remainedGDD,remainedDay

def barPlotCluster2023(df,obslabel,plot=True):
    if plot:
        # plt.figure(figsize=(10,8))
        fig, axs = plt.subplots(2,2,figsize = (10,8),gridspec_kw={'hspace': 0.3})
        axx = axs.flatten(order='C')
    auxText = ['(a) 0N','(b) 50N','(c) 100N','(d) 150N']
    pheno_stats = []
    for i,s in enumerate([0,5,10,15]):
        sampleIndex = [str(t) for t in range(s,s+5)]     
        data = np.array(df[sampleIndex])
        pheno_stat = [len(np.where(data==(i))[0]) for i in range(len(obslabel))]
        if plot:
            ax=axx[i]
            ax.bar(obslabel, pheno_stat,color = 'lightblue')
            ax.tick_params(axis='x', rotation=-30)
            ax.set_ylabel('Fruits number', fontsize = 16)
           
            ax.text(0.05, 0.9, auxText[i], transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=18)
        pheno_stats.append(pheno_stat)

    tmp = np.sum(np.array(pheno_stats),axis=0) 
    percentPheno = tmp[:-1]/np.sum(tmp[:-1]) # discard overripe
    percentOverripe = tmp[-1]/np.sum(tmp[:-1]) 
    if plot:
        fig, ax = plt.subplots(1,figsize = (10,1.5))
        # plot stacked bar plot
        height = 0.3       # the width of the bars: can also be len(x) sequence  
        ax.barh([''], percentPheno[0], height=height, label=obslabel[0],
                color = 'yellow')
        ax.barh([''], percentPheno[1], height=height,left=percentPheno[0], label=obslabel[1],
                color = 'green')
        
        ax.barh([''], percentPheno[2], height=height, left=np.sum(percentPheno[:2]),
               label=obslabel[2], color = 'limegreen')
        ax.barh([''], percentPheno[3], height=height, left=np.sum(percentPheno[:3]),
                label=obslabel[3],color ='mintcream')
        ax.barh([''], percentPheno[4], height=height, left=np.sum(percentPheno[:4]), 
                label=obslabel[4],color ='pink')
        ax.barh([''], percentPheno[5], height=height, left=np.sum(percentPheno[:5]), 
                label=obslabel[5],color ='red')
        # ax.barh([''], percentOverripe, height=height, left=np.sum(percentPheno[:6]), 
        #         label=obslabel[6],color ='indianred', hatch='//')
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1],['0%', '20%', '40%','60%','80%','100%'])
        t = ax.legend(bbox_to_anchor=(1.01, -0.3),frameon = False,ncol =6)
        for handle in t.legendHandles:
            handle.set_linewidth(1.0)  # Set the width of the handle's frame line
            handle.set_edgecolor('black')  # Set the color of the handle's frame line
        # fig.text(0.1, 0.95, '(e)',fontsize=18)
        # ax.set_xlabel("Std of $\mu_t'$ for different years (Bu/Acre)")
    return percentPheno

def get_paraList(ensembleN = 1000, mode = 'diameter'):
       
    # mode = 'length'
    getT = getTrajectories()
    popt, popt_up, popt_d, aux, fig = getT.fitBoundary(mode = mode, per='mannul',plot=True)
    [t, x, y, weather, trajectarys,_] = aux
    # perturbing the growth curve 
    paraList,_ = samplingPara(popt,popt_up,popt_d,t,ensembleN = ensembleN, mode=mode)
    # plt.xlabel(xlabel,fontsize=15)
    plt.ylabel('Fruit %s (mm)'%mode,fontsize=15)
    plt.ylim([0,40])
    plt.legend(edgecolor = 'w',facecolor='w',fontsize=10,loc=4)
    return paraList

if __name__ == '__main__':
    paraList = get_paraList(ensembleN = 1000, mode = 'diameter')

