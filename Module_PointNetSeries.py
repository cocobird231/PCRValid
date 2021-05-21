# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:35:53 2021

@author: User
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#############################################################
#                           PointNet
#############################################################
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetFeat, self).__init__()
        self.stn = STNkd(k=3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[1]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, feat = None, k=40, retGlobFeat = False):
        super(PointNetCls, self).__init__()
        self.retGlobF = retGlobFeat
        self.features = PointNetFeat(True, True) if (not feat) else feat
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)
        x, trans, trans_feat = self.features(x)
        globF = x.contiguous()
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        if (self.retGlobF) : return F.log_softmax(x, dim=1), globF
        else : return F.log_softmax(x, dim=1)


class PointNetComp(nn.Module):
    def __init__(self, k=40, feature_transform=True):
        super(PointNetComp, self).__init__()
        self.features = PointNetFeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, x2):
        x = x.transpose(2, 1)
        x2 = x2.transpose(2, 1)
        x, trans, trans_feat = self.features(x)
        x2, trans, trans_feat = self.features(x2)
        globF = x.contiguous()
        globF2 = x2.contiguous()
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), globF, globF2


#############################################################
#                           DGCNN
#############################################################
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNNFeat(nn.Module):
    def __init__(self, emb_dims = 512, k = 20):
        super(DGCNNFeat, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
    
    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        return x.squeeze()


class DGCNN(nn.Module):
    def __init__(self, feat = None, emb_dims = 512, dp = 0.3, output_channels=40, retGlobFeat = False):
        super(DGCNN, self).__init__()
        self.retGlobF = retGlobFeat
        self.features = DGCNNFeat(emb_dims, k = 20) if (not feat) else feat
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dp)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dp)
        self.linear3 = nn.Linear(256, output_channels)
    
    def forward(self, x):
        x = self.features(x)
        globF = x.contiguous()
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        if (self.retGlobF) : return F.log_softmax(x, dim=1), globF
        else : return F.log_softmax(x, dim=1)


#############################################################
#                           PointNet++
#############################################################
import os
import sys
sys.path.append(os.path.join('/home/wei/Desktop/votenet2', 'pointnet2'))
# from pointnet2_modules import PointnetSAModule

class PointNet2Comp(nn.Module):
    def __init__(self, input_feat_dim = 0, k = 40):
        super().__init__()
        self.sa1 = PointnetSAModule(npoint=512, 
                                    radius=0.2, 
                                    nsample=64, 
                                    mlp=[input_feat_dim, 64, 128])
        
        self.sa2 = PointnetSAModule(npoint=128, 
                                    radius=0.4, 
                                    nsample=64, 
                                    mlp=[128, 128, 256])
        
        self.sa3 = PointnetSAModule(mlp=[256, 512, 1024])
        
        self.fc_layer = nn.Sequential(nn.Linear(1024, 512, bias=False), 
                                      nn.BatchNorm1d(512), 
                                      nn.ReLU(True), 
                                      nn.Linear(512, 256, bias=False), 
                                      nn.BatchNorm1d(256), 
                                      nn.ReLU(True), 
                                      nn.Dropout(0.5), 
                                      nn.Linear(256, k))
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features
        
    def forward(self, x, x2):
        x_xyz, x_feat = self._break_up_pc(x)
        x_xyz, x_feat = self.sa1(x_xyz, x_feat)
        x_xyz, x_feat = self.sa2(x_xyz, x_feat)
        x_xyz, x_feat = self.sa3(x_xyz, x_feat)
        globF = x_feat.contiguous().squeeze(-1)
        x_xyz, x_feat = self._break_up_pc(x2)
        x_xyz, x_feat = self.sa1(x_xyz, x_feat)
        x_xyz, x_feat = self.sa2(x_xyz, x_feat)
        x_xyz, x_feat = self.sa3(x_xyz, x_feat)
        globF2 = x_feat.contiguous().squeeze(-1)
        return F.log_softmax(self.fc_layer(globF), dim=1), globF, globF2


class PointNet2Feat(nn.Module):
    def __init__(self, input_feat_dim = 0):
        super().__init__()
        self.sa1 = PointnetSAModule(npoint=512, 
                                    radius=0.2, 
                                    nsample=64, 
                                    mlp=[input_feat_dim, 64, 128])
        
        self.sa2 = PointnetSAModule(npoint=128, 
                                    radius=0.4, 
                                    nsample=64, 
                                    mlp=[128, 128, 256])
        
        self.sa3 = PointnetSAModule(mlp=[256, 512, 1024])
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features
        
    def forward(self, x):
        x_xyz, x_feat = self._break_up_pc(x)
        x_xyz, x_feat = self.sa1(x_xyz, x_feat)
        x_xyz, x_feat = self.sa2(x_xyz, x_feat)
        x_xyz, x_feat = self.sa3(x_xyz, x_feat)
        return x_feat.squeeze(-1)


class PointNet2Cls(nn.Module):
    def __init__(self, pn2Feat = None, catSize = 40, retGlobFeat = False):
        super().__init__()
        self.retGlobF = retGlobFeat
        self.features = PointNet2Feat() if (not pn2Feat) else pn2Feat
        self.fc_layer = nn.Sequential(nn.Linear(1024, 512, bias=False), 
                              nn.BatchNorm1d(512), 
                              nn.ReLU(True), 
                              nn.Linear(512, 256, bias=False), 
                              nn.BatchNorm1d(256), 
                              nn.ReLU(True), 
                              nn.Dropout(0.5), 
                              nn.Linear(256, catSize))
    
    def forward(self, x):
        globFeat = self.features(x)
        if (self.retGlobF) : return F.log_softmax(self.fc_layer(globFeat), dim=1), globFeat
        else : return F.log_softmax(self.fc_layer(globFeat), dim=1)


if __name__ == '__main__':
    net = PointNet2Comp()
    print(net)