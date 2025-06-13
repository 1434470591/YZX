import torch
import hdf5storage
import numpy as np
import pandas as pd
import torch.utils.data as data
from numpy import random
from einops import rearrange
from utils.tools import StandardScaler
# from utils.tools import LoadBatch_ofdm, noise

class SJTU(data.Dataset):
    def __init__(self, args, flag="train"):
        self.args = args
        self.flag = flag

        assert flag in [
            "train",
            "vali",
            "test",
        ]

        type_map = {
            "train": 0,
            "vali": 1,
            "test": 2,
        }
        self.set_type = type_map[flag]
        self.slid_step = args.slid_step
        self.prev_len = args.prev_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len 

        self.__load_data__()

    def __load_data__(self):
        """
        TODO: download data
        """
        df_raw = pd.read_csv(self.args.root_path + self.args.data_path, encoding="gbk", index_col=0)
        # df = pd.read_csv("./dataset/" + self.args.data_path, encoding="gbk")
        # columns = df.columns
        df_raw.fillna(df_raw.mean(), inplace=True)
        df = df_raw.values[:, :64]
        if self.args.enhancement:
            magnitude = np.sqrt(df[:, :32] ** 2 + df[:, -32:] ** 2)
            data = np.concatenate((df, magnitude, magnitude ** 2,), axis=1)
        else :
            data = df
        """
        TODO: split train&vali&test
        """
        num_train = int(len(data) * self.args.split_ratio[0])
        num_test = int(len(data) * self.args.split_ratio[2])
        num_vali = len(data) - num_train - num_test
        
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(data)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        """
        TODO: standardization
        """
        if self.args.scale:
            train_data = data[border1s[0] : border2s[0]]
            self.scaler = StandardScaler(train_data)
            data = self.scaler.transform(data=data)

        self.data_x = data[border1: border2]
        self.data_y = data[border1: border2]

        self.data_stamp = [[(i + j + 1) for i in range(data.shape[1])] for j in range(data.shape[0])]
        # print(self.data_stamp)

    def __getitem__(self, index):
        x_begin = index * self.slid_step
        x_end = x_begin + self.prev_len

        # y_begin = x_end - self.label_len
        # y_end = y_begin + self.label_len + self.pred_len

        y_begin = x_end
        y_end = y_begin + self.pred_len

        prev_x = self.data_x[x_begin: x_end]
        prev_y = self.data_y[y_begin: y_end]
        return prev_x, prev_y

    def __len__(self):
        return len(self.data_x) - self.prev_len -self.pred_len + 1
    
    def invers(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path_r, file_path_t, is_train=1, ir=1, SNR=15, is_U2D=0, is_few=0,
                 train_per=0.9, valid_per=0.1):
        super(Dataset_Pro, self).__init__()
        self.SNR = SNR
        self.ir = ir
        H_his = hdf5storage.loadmat(file_path_r)['H_U_his_train']  # v,b,l,k,a,b,c
        if is_U2D:
            H_pre = hdf5storage.loadmat(file_path_t)["H_D_pre_train"]  # v,b,l,k,a,b,c
        else:
            H_pre = hdf5storage.loadmat(file_path_t)["H_U_pre_train"]  # v,b,l,k,a,b,c
        # print(H_his.shape, H_pre.shape)

        batch = H_pre.shape[1]
        if is_train:
            H_his = H_his[:, :int(train_per * batch), ...]
            H_pre = H_pre[:, :int(train_per * batch), ...]
        else:
            H_his = H_his[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_pre = H_pre[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
        H_his = rearrange(H_his, 'v n L k a b c -> (v n) L (k a b c)')
        H_pre = rearrange(H_pre, 'v n L k a b c -> (v n) L (k a b c)')

        B, prev_len, mul = H_his.shape
        _, pred_len, mul = H_pre.shape
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.seq_len = pred_len + prev_len

        dt_all = np.concatenate((H_his, H_pre), axis=1)
        np.random.shuffle(dt_all)
        H_his = dt_all[:, :prev_len, ...]
        H_pre = dt_all[:, -pred_len:, ...]
        for i in range(B):
            H_his[i, ...] = noise(H_his[i, ...], random.rand() * 15 + 5.0)
            H_pre[i, ...] = noise(H_pre[i, ...], random.rand() * 15 + 5.0)
        std = np.sqrt(np.std(np.abs(H_his) ** 2))
        H_his = H_his / std
        H_pre = H_pre / std
        H_pre = LoadBatch_ofdm(H_pre)
        H_his = LoadBatch_ofdm(H_his)
        # 10%小样本
        if is_few == 1:
            H_pre = H_pre[::10, ...]
            H_his = H_his[::10, ...]
        self.pred = H_pre  # b,16,(48*2)
        self.prev = H_his  # b,4,(48*2)

    def __getitem__(self, index):
        return self.prev[index, :].float(), \
               self.pred[index, :].float()

    def __len__(self):
        return self.pred.shape[0]
    
def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


def LoadBatch_ofdm_2(H):
    # H: B,T,K,mul     [tensor complex]
    # out:B,T,K,mul*2  [tensor real]
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm_1(H):
    # H: B,T,mul     [tensor complex]
    # out:B,T,mul*2  [tensor real]
    B, T, mul = H.shape
    H_real = np.zeros([B, T, mul, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm(H, num=32):
    # H: B,T,mul             [tensor complex]
    # out:B*num,T,mul*2/num  [tensor real]
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) ->(b a) t k', a=num)
    H_real = np.zeros([B * num, T, mul // num, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B * num, T, mul // num * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def Transform_TDD_FDD(H, Nt=4, Nr=4):
    # H: B,T,mul    [tensor real]
    # out:B',Nt,Nr  [tensor complex]
    H = H.reshape(-1, Nt, Nr, 2)
    H_real = H[..., 0]
    H_imag = H[..., 1]
    out = torch.complex(H_real, H_imag)
    return out
