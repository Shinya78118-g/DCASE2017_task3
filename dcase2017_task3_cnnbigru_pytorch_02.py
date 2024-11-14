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

# read argument
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
    # model training and network parameters
    'nepoch': 500,  # エポック数を指定
    'nbatch': 32,  # バッチサイズを指定
    'hidden_size': 32,  # GRUの隠れ層サイズを設定
    'output_size': len(eventname),  # 出力サイズは音声イベントのクラス数に一致
    'slen': 256,
    'fdim': 40,
    'nevent': len(eventname),
    'thresmode': 'fixed',        # しきい値モード（仮設定）
    'threshold': 0.5,            # 固定しきい値
    'startthres': 0.1,           # 開始しきい値（adaptiveモード用）
    'endthres': 0.9,             # 終了しきい値（adaptiveモード用）
    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu")
}

# define network structure
class CNNBiGRU(nn.Module):
    def __init__(self, params, device='cuda:0'):
        super(CNNBiGRU, self).__init__()
        self.params = params
        self.device = device

        # CNN層を追加（フィルタ数、カーネルサイズ、パディング、バッチ正規化、ドロップアウト）
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)  # 入力チャネルが1、出力チャネルが128
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))  # 周波数軸でのプーリング
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(0.5)

        # GRU層の定義
        self.gru1 = nn.GRU(256, params['hidden_size'], batch_first=True, bidirectional=True)
        self.dropout4 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(params['hidden_size'] * 2, params['hidden_size'], batch_first=True, bidirectional=True)
        self.dropout5 = nn.Dropout(0.5)

        # Time Distributed Dense層の定義
        self.fc1 = nn.Linear(params['hidden_size'] * 2, 16)
        self.dropout6 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, params['output_size'])

    def forward(self, x0):
        x0 = x0.unsqueeze(1)  # チャンネル次元を追加して [nbatch, 1, T, fdim] の形にする

        # CNN層の処理
        x1 = F.relu(self.bn1(self.conv1(x0)))
        x1 = self.pool1(x1)
        x1 = self.dropout1(x1)

        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.pool2(x2)
        x2 = self.dropout2(x2)

        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = self.pool3(x3)
        x3 = self.dropout3(x3)

        # GRUに入力するための形に変換
        x3 = x3.permute(0, 2, 1, 3).contiguous()
        x3 = x3.view(x3.size(0), x3.size(1), -1)  # [nbatch, T, feature_dim]

        # GRU層の処理
        x4, _ = self.gru1(x3)
        x4 = self.dropout4(x4)
        x5, _ = self.gru2(x4)
        x5 = self.dropout5(x5)

        # Time Distributed Dense層の処理
        x6 = F.relu(self.fc1(x5))
        x6 = self.dropout6(x6)
        x7 = torch.sigmoid(self.fc2(x6))

        return x7


def main():

    if params['mode'] == 'train':

        # fix seed for random functions
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # load dataset and set dataloader
        traindata = SoundDataset.SEDDataset(traindfn, trainlfn, params, train=True)
        testdata = SoundDataset.SEDDataset(testdfn, testlfn, params, train=False)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=params['nbatch'], shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=params['nbatch'], shuffle=False, drop_last=True)

        # define network structure
        model = CNNBiGRU(params).to(params['device'])

        # set loss function and optimizer
        criterion = nn.BCELoss()  # バイナリクロスエントロピー損失
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # model training
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop_patience = 100

        for epoch in range(params['nepoch']):
            model.train()
            train_loss = 0.0
            for data, labels in train_loader:
                data, labels = data.to(params['device']), labels.to(params['device']).float()
                
                # フォワードとバックプロパゲーション
                outputs = model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')

            # calculate loss for evaluate dataset
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(params['device']), labels.to(params['device']).float()
                    outputs = model(data)
                    val_loss += criterion(outputs, labels).item()

            val_loss /= len(test_loader)
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # save model and params
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, resname + '.pth')

    elif params['mode'] == 'test' or params['mode'] == 'eval' or params['mode'] == 'evaluate':

        # load dataset and set dataloader
        testdata = SoundDataset.SEDDataset(testdfn, testlfn, params, train=False)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=params['nbatch'], shuffle=False, drop_last=True)

        # define network structure
        model = CNNBiGRU(params).to(params['device'])

        # load model
        checkpoint = torch.load(resname + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # オプティマイザの再定義
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # calculate sound event labels & their boundaries
        model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(params['device']), labels.to(params['device']).float()
                outputs = model(data)
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # 全てのバッチを連結して1つの配列にし、タイムステップの次元をイベント次元に変換する
        all_outputs = np.vstack(all_outputs)  # [num_batches * batch_size, T, num_events] -> [num_samples, num_events]
        all_labels = np.vstack(all_labels)

        # reshapeして各イベントごとのラベルと出力をイベントレベルにする
        if len(all_outputs.shape) == 3:
            all_outputs = np.sum(all_outputs, axis=1)  # タイムステップを集約
        if len(all_labels.shape) == 3:
            all_labels = np.sum(all_labels, axis=1)  # タイムステップを集約

        # ビット演算ができるようにデータ型を変換
        all_outputs = all_outputs.astype(int)
        all_labels = all_labels.astype(int)

        # SEDresult クラスを使用してイベント境界の計算を行う
        evaluator_params = {
            'thresmode': 'fixed',  
            'threshold': 0.5,
            'startthres': 0.1,
            'endthres': 0.9,
            'nevent': len(eventname)
        }
        result = evaluator.SEDresult(all_outputs, all_labels, evaluator_params)
        result.sed_evaluation(plotflag=True, saveflag=True, path=resname + '/' + resname)

if __name__ == '__main__':
    main()
