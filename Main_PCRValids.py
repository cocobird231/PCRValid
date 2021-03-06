# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:00:44 2021

@author: User
"""

import os
import time
import copy
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from Module_Utils import textIO, ICPIter
from Module_DCP import Identity
from Module_PointNetLK import PointLK
from Module_PointNetSeries import PointNet2Feat
from utils.pointnet import PointNet_features# Remove while re-train
from utils.dcp_old import DCP, DCPProp# Remove while re-train
from Module_ModelNet40Series_DataLoader import RegistrationValidDataset

import argparse

modelList = ['pointnetlk', 'pointnetlk2', 'dcp', 'fgr', 'icp']

def getParser():
    parser = argparse.ArgumentParser(description='PCRValidation')
    parser.add_argument('-d', '--dataset', default='D:\\Datasets\\ModelNet40_VALID_1024_2', required=False, type=str, 
                        metavar='PATH', help='path to the valitation dataset')
    parser.add_argument('-m', '--model', default='dcp', required=False, type=str, 
                        metavar='MODEL', choices=modelList, help='Model validation')
    parser.add_argument('-p', '--modelPath', default='models/model_DCP_PN_V_E1000_D2.t7', required=False, type=str, 
                        metavar='PATH', help='Model path (feat)')
    parser.add_argument('-q', '--modelPath2', default='models/PointNetLK/pointlk_model_best.pth', required=False, type=str, 
                        metavar='PATH', help='Model path (lk)')
    parser.add_argument('-i', '--iter', type=int, 
                        default=None, help='ICP iteration times')
    parser.add_argument('-l', '--saveLogDir', default='result', required=False, type=str, 
                        metavar='PATH', help='Log saving path')
    parser.add_argument('--logName', default=None, type=str, metavar='NAME', help='Log file name (default: model name)')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda')
    return parser.parse_args()

def initEnv(args):
    try:
        if (not os.path.exists(args.saveLogDir)):
            os.mkdir(args.saveLogDir)
        if (not os.path.exists(args.dataset)):
            raise 'Dataset path error'
        if (not os.path.exists(args.modelPath)):
            raise 'Model path error'
        if (args.model == 'pointnetlk' or args.model == 'pointnetlk2'):
            if (not os.path.exists(args.modelPath2)):
                raise 'Model path 2 error'
        if (args.model not in modelList):
            raise 'Selected model error choices:{}'.format(modelList)
        args.logName = 'log_{}.txt'.format(args.model)
        textLog = textIO(args)
        textLog.writeLog(time.ctime())
        textLog.writeLog(args.__str__())
        return textLog
    except:
        raise 'Unexpected error'


def RunValidation(net, testLoader, textLog, args):
    net.eval()
    avgAngMAE = 0
    avgAngRMSE = 0
    avgTransMAE = 0
    avgTransRMSE = 0
    totalRankDict = {'Rank 1' : 0, 'Rank 5' : 0, 'Rank 10' : 0, 'Rank 20' : 0, 'Rank 30' : 0, 'Out of Rank' : 0}
    ICP_PC = o3d.geometry.PointCloud()
    tempPC = o3d.geometry.PointCloud()
    
    sT = time.clock()
    for tmpPCD, tarPCD, rotMat, transVec in testLoader:
        # Init point cloud normals
        if (args.iter or args.model == 'fgr'):
            ICP_PC.points = o3d.utility.Vector3dVector(tmpPCD)
            tempPC.points = o3d.utility.Vector3dVector(tarPCD)
            ICP_PC.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
            tempPC.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
        
        tmpPCD = torch.tensor(tmpPCD).unsqueeze(0)
        tarPCD = torch.tensor(tarPCD).unsqueeze(0)
        if (args.cuda) : tmpPCD = tmpPCD.cuda()
        if (args.cuda) : tarPCD = tarPCD.cuda()
        
        ## Init output Rigid
        outRot = np.eye(3)
        outTrans = np.asarray([0, 0, 0])
        outTransMat = np.eye(4)
        outAng = np.asarray([0, 0, 0])
        
        if (args.model == 'pointnetlk' or args.model == 'pointnetlk2'):
            _ = PointLK.do_forward(net, tmpPCD, tarPCD, maxiter=20)
            est_g = net.g.detach()
            if (args.cuda) : est_g = est_g.cpu()
            estTransform = est_g.numpy().squeeze()
            # pcd2 to pcd1
            invRot = estTransform[:3,:3].T
            invTrans = -invRot.dot(estTransform[:3, 3])
            invTransform = np.block([[invRot, invTrans.reshape(3, -1)], [np.eye(4)[-1]]])
            invAng = R.from_matrix(invRot).as_euler('xyz', True)
            
            outRot = invRot
            outTrans = invTrans
            outTransMat = invTransform
            outAng = invAng
            
            
        elif (args.model == 'dcp'):
            rot_ab_pred, trans_ab_pred, _, _ = net(tmpPCD, tarPCD)
            if (args.cuda) : rot_ab_pred = rot_ab_pred.cpu()
            if (args.cuda) : trans_ab_pred = trans_ab_pred.cpu()
            
            outRot = rot_ab_pred.detach().numpy().squeeze()
            outTrans = trans_ab_pred.detach().numpy().squeeze()
            outTransMat = np.block([[outRot, outTrans.reshape(3, -1)], [np.eye(4)[-1]]])
            outAng = R.from_matrix(outRot).as_euler('xyz', True)
            
            
        elif (args.model == 'fgr'):
            ICP_PC_FPFH = o3d.pipelines.registration.compute_fpfh_feature(
                        ICP_PC, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
            tempPC_FPFH = o3d.pipelines.registration.compute_fpfh_feature(
                        tempPC, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))
            FGR = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
                        copy.deepcopy(ICP_PC), copy.deepcopy(tempPC), ICP_PC_FPFH, tempPC_FPFH)
            FGR_TRANSFORM = FGR.transformation
            
            outRot = FGR_TRANSFORM[:3, :3]
            outTrans = FGR_TRANSFORM[:3, 3]
            outTransMat = FGR_TRANSFORM
            outAng = R.from_matrix(outRot).as_euler('xyz', True)
            
            
        if (args.model == 'icp' or args.iter):
            assert (args.iter != None), 'ICP method must assign --iter value'
            ICP_TRANSFORM = ICPIter(ICP_PC, tempPC, outTransMat, args.iter)
            ICP_TRANSFORM = ICP_TRANSFORM.transformation
            
            outRot = ICP_TRANSFORM[:3, :3]
            outTrans = ICP_TRANSFORM[:3, 3]
            outTransMat = ICP_TRANSFORM
            outAng = R.from_matrix(outRot).as_euler('xyz', True)
        
        gtAng = R.from_matrix(rotMat).as_euler('xyz', True)
        
        # print('GT Ang  :', gtAng)
        # print('Pred Ang:', outAng)
        # print('GT Trans  :', transVec)
        # print('Pred Trans:', outTrans)
        # # Visualization
        # if (args.cuda) : tmpPCD = tmpPCD.detach().cpu()
        # if (args.cuda) : tarPCD = tarPCD.detach().cpu()
        # tmpPCD = tmpPCD.numpy().squeeze()
        # tarPCD = tarPCD.numpy().squeeze()
        # # TmpPCD
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(tmpPCD)
        # pcd1.paint_uniform_color([1, 0, 0])
        # # TarPCD
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(tarPCD)
        # pcd2.paint_uniform_color([0, 1, 0])
        # # PredPCD
        # pcd3 = copy.deepcopy(pcd1)
        # pcd3.paint_uniform_color([0, 0, 1])
        # pcd3.transform(outTransMat)
        # # GTPCD
        # pcd4 = copy.deepcopy(pcd1)
        # pcd4.paint_uniform_color([0, 0.5, 1])
        # pcd4.transform(np.block([[rotMat, transVec.reshape(3, -1)], [np.eye(4)[-1]]]))
        # # Visualize
        # o3d.visualization.draw_geometries([pcd2, pcd3, pcd4], window_name = 'Result')
        
        # Rank evaluation
        AngMAE = np.mean(np.abs(outAng - gtAng))
        avgAngMAE += AngMAE
        AngRMSE = np.sqrt(np.mean(np.square(outAng - gtAng)))
        avgAngRMSE += AngRMSE
        
        avgTransMAE += np.mean(np.abs(outTrans - transVec))
        avgTransRMSE += np.sqrt(np.mean(np.square(outTrans - transVec)))
        
        if (AngRMSE <= 1) : totalRankDict['Rank 1'] += 1
        if (AngRMSE <= 5) : totalRankDict['Rank 5'] += 1
        if (AngRMSE <= 10) : totalRankDict['Rank 10'] += 1
        if (AngRMSE <= 20) : totalRankDict['Rank 20'] += 1
        if (AngRMSE <= 30) : totalRankDict['Rank 30'] += 1
        if (AngRMSE > 30) : totalRankDict['Out of Rank'] += 1
        
    textLog.writeLog('Total use {} sec'.format(time.clock() - sT))
    cnt = 0
    for key in totalRankDict : cnt += totalRankDict[key]
    textLog.writeLog('Average Angle MAE: %f' %(avgAngMAE / cnt))
    textLog.writeLog('Average Angle RMSE: %f' %(avgAngRMSE / cnt))
    textLog.writeLog('Average Trans MAE: %f' %(avgTransMAE / cnt))
    textLog.writeLog('Average Trans RMSE: %f' %(avgTransRMSE / cnt))
    textLog.writeLog('{}'.format(totalRankDict))

if (__name__ == '__main__'):
    args = getParser()
    args.cuda = True if args.cuda and torch.cuda.is_available() else False
    device = 'cuda:0' if args.cuda else 'cpu'
    device = torch.device(device)
    
    textLog = initEnv(args)
    if (args.model == 'pointnetlk'):
        ptnet = PointNet_features()
        ptnet.load_state_dict(torch.load(args.modelPath, map_location='cpu'))
        net = PointLK(ptnet, 1.0e-2)
        net.load_state_dict(torch.load(args.modelPath2, map_location='cpu'))
        net.to(device)
    elif (args.model == 'pointnetlk2'):
        ptnet = PointNet2Feat()
        ptnet.load_state_dict(torch.load(args.modelPath, map_location='cpu'))
        net = PointLK(ptnet, 1.0e-2)
        net.load_state_dict(torch.load(args.modelPath2, map_location='cpu'))
        net.to(device)
    elif (args.model == 'dcp'):
        net = DCP(DCPProp())
        net.load_state_dict(torch.load(args.modelPath, map_location='cpu'))
        net.to(device)
    elif (args.model == 'fgr'):
        net = Identity()
    elif (args.model == 'icp'):
        net = Identity()
    testLoader = RegistrationValidDataset(args.dataset)
    RunValidation(net, testLoader, textLog, args)
    textLog.close()