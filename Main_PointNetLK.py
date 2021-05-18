# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:31:33 2021

@author: User
"""

import os
import csv
import time
import copy
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R

import torch

from Module_Utils import textIO
from Module_Parser import PointNetLKParser
from Module_PointNetLK_DataLoader import ModelNet40_VALID

import ptlk


def RunValidation(net, testLoader, args):
    net.eval()
    for tmpPCD, tarPCD, rotMat, transVec in testLoader:
        tmpPCD = torch.tensor(tmpPCD).unsqueeze(0)
        tarPCD = torch.tensor(tarPCD).unsqueeze(0)
        if (args.cuda) : tmpPCD = tmpPCD.cuda()
        if (args.cuda) : tarPCD = tarPCD.cuda()
        r = ptlk.pointlk.PointLK.do_forward(net, tmpPCD, tarPCD, maxiter=args.max_iter)
        est_g = net.g.detach()
        if (args.cuda) : est_g = est_g.cpu()
        estTransform = est_g.numpy().squeeze()
        
        estAng = R.from_matrix(estTransform[:3, :3]).as_euler('xyz', True)
        gtAng = R.from_matrix(rotMat).as_euler('xyz', True)
        
        print(estAng)
        print(gtAng)
        
        estTrans = estTransform[:3,3]
        print(estTrans)
        print(transVec)
        
        if (args.cuda) : tmpPCD = tmpPCD.detach().cpu()
        if (args.cuda) : tarPCD = tarPCD.detach().cpu()
        tmpPCD = tmpPCD.numpy().squeeze()
        tarPCD = tarPCD.numpy().squeeze()
        
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(tmpPCD)
        pcd1.paint_uniform_color([1, 0, 0])
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(tarPCD)
        pcd2.paint_uniform_color([0, 1, 0])
        
        # pcd2 to pcd1
        invRot = estTransform[:3,:3].T
        invTrans = -invRot.dot(estTransform[:3, 3].reshape(3, -1))
        invTransform = np.block([[invRot, invTrans], [np.eye(4)[-1]]])
        
        pcd3 = copy.deepcopy(pcd1)
        pcd3.paint_uniform_color([0, 0, 1])
        pcd3.transform(invTransform)
        
        pcd4 = copy.deepcopy(pcd1)
        pcd4.paint_uniform_color([0, 0.5, 1])
        pcd4.transform(np.block([[rotMat, transVec.reshape(3, -1)], [np.eye(4)[-1]]]))
        
        o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, pcd4], window_name = 'Result')


def initEnv(args):
    try:
        if (not os.path.exists(args.saveLogDir)):
            os.mkdir(args.saveLogDir)
        if (not os.path.exists(args.dataset)):
            raise 'Dataset path error'
        if (not os.path.exists(args.clsModelPath)):
            raise '--clsModelPath path error'
        if (not os.path.exists(args.lkModelPath)):
            raise '--lkModelPath path error'
        textLog = textIO(args)
        textLog.writeLog(time.ctime())
        return textLog
    except:
        raise 'Unexpected error'


def initDevice(args):
    if (not args.cuda or not torch.cuda.is_available()):
        device = torch.device('cpu')
        args.cuda = False
    elif (torch.device(args.cudaDevice)):
        device = torch.device(args.cudaDevice)
        torch.cuda.set_device(device.index)
    else:
        device = torch.device('cpu')
        args.cuda = False
    return device


if (__name__ == '__main__'):
    args = PointNetLKParser()
    textLog = initEnv(args)
    device = initDevice(args)
    textLog.writeLog(args.__str__())
    testLoader = ModelNet40_VALID(args.dataset)
    ptnet = ptlk.pointnet.PointNet_features(args.dim_k, use_tnet=False, sym_fn=ptlk.pointnet.symfn_max)
    ptnet.load_state_dict(torch.load(args.clsModelPath, map_location='cpu'))
    net = ptlk.pointlk.PointLK(ptnet, args.delta)
    net.load_state_dict(torch.load(args.lkModelPath, map_location='cpu'))
    net.to(device)
    RunValidation(net, testLoader, args)
    textLog.close()