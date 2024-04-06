# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:14:16 2024

@author: yang8460

Split the dataset to training set and test set
"""
import glob,os,shutil
import numpy as np
import json

def datset_split(inputPath, outPath, test_ratio = 0.2):
    dataset = glob.glob('%s/*.jpg'%inputPath)
    Nlist = sorted([t.split('\\')[-1].split('.')[0] for t in dataset])
    test_n = int(len(dataset)*test_ratio)
    np.random.seed(0)
    np.random.shuffle(Nlist)
    
    dataset_train = Nlist[:-test_n]
    dataset_test =  Nlist[-test_n:]
    
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    if not os.path.exists('%s/train'%outPath):
        os.mkdir('%s/train'%outPath)
    for train in dataset_train:
        shutil.copyfile('%s/%s.jpg'%(inputPath,train), 
               '%s/train/%s.jpg'%(outPath,train))
        shutil.copyfile('%s/%s.json'%(inputPath,train), 
                   '%s/train/%s.json'%(outPath,train))
        
            
    if not os.path.exists('%s/test'%outPath):
        os.mkdir('%s/test'%outPath)
    for test in dataset_test:
        shutil.copyfile('%s/%s.jpg'%(inputPath,test), 
               '%s/test/%s.jpg'%(outPath,test))
        shutil.copyfile('%s/%s.json'%(inputPath,test), 
                   '%s/test/%s.json'%(outPath,test))

def dataset_merge(datasetList, outPath):
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    for folder in datasetList:
        jsonList = glob.glob('%s/*.json'%folder)
        IdList = [t.split('\\')[-1].split('.')[0] for t in jsonList]
        
        # copy json
        for j,img_id in zip(jsonList,IdList):

            # copy img
            shutil.copyfile('%s/%s.jpg'%(folder,img_id), '%s/%s.jpg'%(outPath,img_id))
            
            # load json
            with open(j,'r') as f:
                json_tmp = json.load(f)
            # discard img str
            json_tmp['imageData'] = None
            #revise the path
            json_tmp['imagePath'] = '%s.jpg'%(img_id)
            # save json
            with open('%s/%s.json'%(outPath,img_id), 'w') as f:
                json.dump(json_tmp,f, indent=2)
        
        print('processing %s'%folder)
        
if __name__ == '__main__':

    # split the dataset for fruit size CNN
    datset_split(inputPath='strawberry_img_tagged', outPath='strawberry_fruitSize', test_ratio = 0.2)
    
    # merge & split datasets for FasterRCNN
    dataset_merge(datasetList=['strawberry_img_tagged', 'strawberry_fruitSize'], 
                  outPath='strawberry_alldata')
    datset_split(inputPath='strawberry_alldata', outPath='strawberry_fasterRCNN', test_ratio = 0.2)

    
