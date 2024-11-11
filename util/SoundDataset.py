import csv
import numpy as np
import torch
from torch.utils.data import Dataset

class SEDDataset(Dataset):
    def __init__(self, datafn, labelfn, params, transform=None, train=True):
        self.transform = transform
        self.train = train

        # load dataset
        data = []; label = []
        with open(datafn,'r') as datafid, open(labelfn,'r') as labelfid:
            datareader = csv.reader(datafid)
            labelreader = csv.reader(labelfid)
            for rowd,rowl in zip(datareader,labelreader):
                data.append(rowd), label.append(rowl)

            data = np.array(data,dtype=float)
            label = np.array(label,dtype=float)

        # reshape data
        data = data.reshape([-1,params['slen'],params['fdim']])
        label = label.reshape([-1,params['slen'],params['nevent']])

        self.data = data
        self.label = label
        self.datanum = len(data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx],dtype=torch.float32)
        label = torch.tensor(self.label[idx],dtype=torch.float32)

        return data, label
