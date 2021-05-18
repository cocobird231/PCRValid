# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:19:15 2021

@author: User
"""

import os
import csv
import glob
import h5py
import numpy as np
import open3d as o3d

from torch.utils.data import Dataset

from Module_Utils import jitter_pointcloud, scaling_pointCloud, rotate_pointCloud, ModelUtils, Rigid

ModelNet40H5ReturnTypeList = ['cls', 'glob2', 'triplet', 'lk']

def GetModelNet40H5ReturnType(args):
    if (args.L1Loss or args.L2Loss) : return 'glob2'
    elif (args.tripletMg) : return 'triplet'
    else : return 'cls'

#############################################################
#                       Training Dataset
#############################################################
# Support ModelSelector PointNetLK
class ModelNet40H5(Dataset):# modelnet40_ply_hdf5_2048 is a 2048 points pcd normalized into unit sphere.
    def __init__(self, DIR_PATH : str, dataPartition = 'None', 
                 tmpPointNum = 1024, srcPointNum = 1024, 
                 gaussianNoise = True, randView = False, scaling = False, 
                 angleRange = 90, translationRange = 0.5, scalingRange = 0.2, retType = 'cls'):
        
        self.data, self.label = self.load_data(DIR_PATH, dataPartition)
        
        self.tmpPointNum = tmpPointNum
        self.srcPointNum = srcPointNum
        self.gaussianNoise = gaussianNoise
        self.randView = randView
        self.scaling = scaling
        self.angleRange = angleRange
        self.translationRange = translationRange
        self.scalingRange = scalingRange
        
        # cls:      return pc1, label (pc1:Nx3)
        # glob2:    return pc1, pc2, label (pc2 is nearly equivalent to pc1)
        # triplet:  return pc1, pc2, pc3, label (pc3 is different between pc1 and pc2, but same label)
        assert retType in ModelNet40H5ReturnTypeList, 'retType must in {}'.format(ModelNet40H5ReturnTypeList)
        self.retType = retType
        self.pc2F = True if (self.retType == 'glob2' or self.retType == 'triplet') else False
        self.pc3F = True if (self.retType == 'triplet') else False
        if (self.retType == 'triplet') : self.catDataIdxDict = self.getCatData()
    
    def load_data(self, DIR_PATH, dataPartition):
        all_data = []
        all_label = []
        dataNamePattern = '/ply_data*.h5'
        if (dataPartition != 'None'):
            dataNamePattern = ('/ply_data_%s*.h5' %dataPartition)
        print(dataNamePattern)
        for h5_name in glob.glob(DIR_PATH + dataNamePattern):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label
    
    def getCatData(self):
        catList = self.label.T.squeeze().tolist()
        catSet = set(catList)
        catDataIdxDict = dict()
        for cat in catSet : catDataIdxDict[cat] = []
        for i, label in enumerate(catList) : catDataIdxDict[label].append(i)
        return catDataIdxDict
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        rigidAB = Rigid(getRandF = True, randAngRange = self.angleRange, randTransRange = self.translationRange)
        
        pc = self.data[item]
        pc1 = np.random.permutation(pc)[:self.srcPointNum]
        if (self.pc2F):
            pc2 = np.random.permutation(pc)[:self.tmpPointNum]
            pc2 = rotate_pointCloud(pc2, rigidAB)
        if (self.pc3F):
            # Select label index that differs with pc1 but as same category
            catDataIdxList = self.catDataIdxDict[self.label[item].item()]
            _selec = item
            while (_selec == item) : _selec = np.random.choice(catDataIdxList)
            pc3 = self.data[_selec]
            pc3 = np.random.permutation(pc3)[:self.tmpPointNum]
            pc3 = rotate_pointCloud(pc3)
        # Additional operation
        if (self.gaussianNoise):
            pc1 = jitter_pointcloud(pc1)
            if (self.pc2F) : pc2 = jitter_pointcloud(pc2)
            if (self.pc3F) : pc3 = jitter_pointcloud(pc3)
        if (self.scaling):
            pc1 = scaling_pointCloud(pc1)
            if (self.pc2F) : pc2 = scaling_pointCloud(pc2)
            if (self.pc3F) : pc3 = scaling_pointCloud(pc3)
        # Output pc1, pc2, pc3: N x 3
        if (self.retType == 'cls'):
            return pc1.astype('float32'), self.label[item]
        elif (self.retType == 'glob2'):
            return pc1.astype('float32'), pc2.astype('float32'), self.label[item]
        elif (self.retType == 'triplet'):
            return pc1.astype('float32'), pc2.astype('float32'), pc3.astype('float32'), self.label[item]
        elif (self.retType == 'lk'):
            return pc1.astype('float32'), pc2.astype('float32'), rigidAB.getTransMat().astype('float32')


#############################################################
#               ModelSelector Validation Dataset
#############################################################
# Support ModelSelector
class ModelSelectorValidDataset():
    def __init__(self, VALID_DIR : str, specCatList = []):# -> VALID_DIR: Directory path for ModelNet40_ModelSelector_VALID
        self.srcPCDList, self.srcPathList, self.ansPathList, self.catList, self.catPCDsDict = self.getModelsFromModelSelectorVALIDDataset(VALID_DIR, specCatList)
        assert (len(self.srcPCDList) == len(self.srcPathList) == len(self.ansPathList) == len(self.catList)), 'Data length error'
        self.size = len(self.srcPCDList)
    
    def getModelsFromModelSelectorVALIDDataset(self, VALID_DIR : str, specCatList = []):
        srcdir = os.path.join(VALID_DIR, '_src')
        catList = []
        srcPCDList = []# Stored each testing srcPCD
        catPCDsDict = dict()# Each category's PCD: {'cat' : [ModelUtils(PCD, cat, path),...,ModelUtils(PCD, cat, path)]}
        srcPathList = []# Path for srcPCD in /_src directory
        ansPathList = []# Path for srcPCD in /'cat' directory
        with open(os.path.join(srcdir, '_Association.csv'), 'r', encoding = 'utf-8', newline = '') as f:
            csvReader = csv.reader(f)
            for row in csvReader:
                if (len(specCatList) > 0):
                    if (not row[0] in specCatList) : continue
                catList.append(row[0])
                srcPathList.append(row[1])
                ansPathList.append(row[2])
                srcPCDList.append(ModelUtils(np.asarray(o3d.io.read_point_cloud(os.path.join(srcdir, row[1])).points), row[0], row[1]))
        catSet = set(catList)
        for cat in catSet:
            fileNameList = WalkModelNet40ByCatName(VALID_DIR, cat, '.pcd', retFile = 'name')
            modelList = []
            for name in fileNameList:
                filePath = os.path.join(VALID_DIR, cat, name)
                modelList.append(ModelUtils(np.asarray(o3d.io.read_point_cloud(filePath).points), cat, name))
            catPCDsDict[cat] = modelList
        return srcPCDList, srcPathList, ansPathList, catList, catPCDsDict
    
    def getModelListByCat(self, category : str):
        return self.catPCDsDict[category]
    
    def getAllCatModelDict(self):
        return self.catPCDsDict
    
    def __iter__(self):
        self.cnt = 0
        return self
    
    def __next__(self):
        if (self.cnt < self.size):
            _temp = self.cnt
            self.cnt += 1
            return self.srcPCDList[_temp], self.getModelListByCat(self.srcPCDList[_temp].label), self.ansPathList[_temp]
        else:
            raise StopIteration


#############################################################
#               Registration Validation Dataset
#############################################################
# Support PointNetLK DCP ICP
class RegistrationValidDataset(Dataset):
    def __init__(self, DIR_PATH : str):
        self.tmpPCDList, self.tarPCDList, self.rotMatList, self.transVecList = self.loadDataFromCSV(DIR_PATH)
        
    def loadDataFromCSV(self, DIR_PATH : str):
        tmpPCDList = []
        tarPCDList = []
        rotMatList = []
        transVecList = []
        with open(os.path.join(DIR_PATH, 'rigids.csv'), 'r', encoding = 'utf-8', newline = '') as f:
            csvReader = csv.reader(f)
            for i, row in enumerate(csvReader):
                if (i == 0) : continue
                tmpPCD = o3d.io.read_point_cloud(os.path.join(DIR_PATH, 'template', '%s.pcd' %row[0]))
                tarPCD = o3d.io.read_point_cloud(os.path.join(DIR_PATH, 'target', '%s.pcd' %row[1]))
                rotStr = row[2].replace('[', '').replace(']', '').strip().split('\n')
                rotMat = np.asarray([[float(itemStr) for itemStr in rotStr[i].split()] for i in range(3)])
                transStr = row[3].replace('[', '').replace(']', '').strip().split()
                transVec = np.asarray([float(item) for item in transStr])

                tmpPCDList.append(np.asarray(tmpPCD.points))
                tarPCDList.append(np.asarray(tarPCD.points))
                rotMatList.append(rotMat)
                transVecList.append(transVec)
        return tmpPCDList, tarPCDList, rotMatList, transVecList
    
    def __len__(self):
        return len(self.tmpPCDList)
    
    def __getitem__(self, item):
        return self.tmpPCDList[item].astype('float32'), self.tarPCDList[item].astype('float32'), \
                self.rotMatList[item].astype('float32'), self.transVecList[item].astype('float32')



#############################################################
#               ModelNet40 Dataset Implementation
#############################################################
def WalkModelNet40ByCatName(DIR_PATH : str, CAT_PATH : str, extName : str = '.off', retFile : str = 'path'):
    assert retFile == 'all' or retFile == 'path' or retFile == 'name'
    filePathList = []
    fileNameList = []
    for dirpath, dirnames, filename in os.walk(os.path.join(DIR_PATH, CAT_PATH)):
        for modelName in filename:
            if (modelName[-len(extName):] == extName):
                filePathList.append(os.path.join(dirpath, modelName))
                fileNameList.append(modelName)
    if (retFile == 'all') : return filePathList, fileNameList
    if (retFile == 'path') : return filePathList
    if (retFile == 'name') : return fileNameList


def WalkModelNet40CatDIR(DIR_PATH : str):
    catList = []
    for dirpath, dirnames, filename in os.walk(DIR_PATH):
        for catdir in dirnames:
            catList.append(catdir)
        break
    return catList



#############################################################
if __name__ == '__main__':

    import sys
    loader = ModelSelectorValidDataset('D:/Datasets/ModelNet40_ModelSelector_VALID')
    cnt = 1
    for srcModelU, catModelUList, pathAns in loader:
        print(srcModelU)
        print(catModelUList)
        print(pathAns)
        cnt += 1
        if (cnt > 5) : break
    sys.exit(0)
    from Module_Parser import ModelSelectorParser
    from torch.utils.data import DataLoader
    args = ModelSelectorParser()
    
    trainLoader = DataLoader(ModelNet40H5(dataPartition='train', DIR_PATH=args.dataset, 
                                          srcPointNum=args.inputPoints, 
                                          tmpPointNum=args.inputPoints, 
                                          gaussianNoise=args.gaussianNoise, 
                                          scaling=args.scaling), 
                            batch_size=1, shuffle=True)
    cnt = 1
    for pc1, pc2, label in trainLoader:
        pc1 = pc1.numpy().squeeze()
        pc2 = pc2.numpy().squeeze()
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1)
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        o3d.visualization.draw_geometries([pcd1, pcd2], window_name = 'Result')
        cnt += 1
        if (cnt > 5) : break