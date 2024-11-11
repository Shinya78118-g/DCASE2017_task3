# -*- coding: utf-8 -*-
import sys, os, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import SoundDataset
from sed_util import evaluator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

eventname = ['brakes squeaking','car','children','large vehicle','people speaking','people walking']

# read argment
args = sys.argv
argc = len(args)

# file name of feature and label data
traindfn = './dataset/train_data_fold' + '{0:02d}'.format(int(sys.argv[2])) + '.csv'
trainlfn = './dataset/train_label_fold' + '{0:02d}'.format(int(sys.argv[2])) + '.csv'
testdfn = './dataset/test_data_fold' + '{0:02d}'.format(int(sys.argv[2])) + '.csv'
testlfn = './dataset/test_label_fold' + '{0:02d}'.format(int(sys.argv[2])) + '.csv'
resname = 'dcase2017_task3_cnnbigru_fold' + '{0:02d}'.format(int(sys.argv[2])) + '_01'

# parameter settings
params = {
    'mode': sys.argv[1],
    'fold': sys.argv[2],
    # model trainig and network parameters
    'nepoch': xxx,
}

# define network structure
class CNNBiGRU(nn.Module):
    #def __init__(self, filter_num=32, filter_size=3, stride=1, hidden_size=100, output_size=6, device='duda:0'):
    def __init__(self, params, device='duda:0'):
        super(CNNBiGRU, self).__init__()
        self.params = params
        self.device = device

    def forward(self,x0): # x0.size = torch.Size([nbatch, slen=256, fdim=40])

        xxx

        return x5


def main():

    if params['mode'] == 'train':

        # fix seed for rand functions

        # load dataset and set dataloader
        traindata = SoundDataset.SEDDataset(traindfn,trainlfn,params,train=True)
        testdata = SoundDataset.SEDDataset(testdfn,testlfn,params,train=False)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=params['nbatch'], shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=params['nbatch'], shuffle=False, drop_last=True)

        # define network structure

        # set loss function and optimizer
 
        # model training

            # calculate loss for evaluate dataset

            # print loss


        # save model and params

    elif params['mode'] == 'test' or params['mode'] == 'eval' or params['mode'] == 'evaluate':

        # load dataset and set dataloader

        # define network structure

        # load model


    # calculate sound event labels & their boundaries

    # calculate SED performance
    result = evaluator.SEDresult(outputs,labels,params)
    result.sed_evaluation(plotflag=True,saveflag=True,path=resname+'/'+resname)

if __name__ == '__main__':
    main()
