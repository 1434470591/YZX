import csv
import os
import time
import warnings
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from matplotlib import pyplot as plt
from thop import profile
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, mkdir
from utils.losses import SE_Loss, NMSELoss

class Exp_SJTU(Exp_Basic):
    def __init__(self, args):
        super(Exp_SJTU, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))

        total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

        return model
    
    def _get_data(self, flag):
        return data_provider(self.args, flag)
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_func='MSE'):
        if loss_func == 'MSE':
            criterion = nn.MSELoss()
        elif loss_func == 'NMSE':
            criterion = NMSELoss()

        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag="vali")
        # test_loader = self._get_data(flag="test")

        # save path
        path = os.path.join(self.args.checkpoints, setting)
        mkdir(path)
        plt.clf()
        visual_train_loss = []
        visual_vali_loss = []
        visual_test_loss = []
        
        train_start_time = time.time()
        train_steps = len(train_loader)
        print(f'Train steps: {train_steps}.')
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss_func)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_loss = []
            self.model.train()
            for iter_count, (batch_x, batch_y) in enumerate(train_loader):
                # clear the gradients
                model_optim.zero_grad()

                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-xl']:
                            outputs = self.model(batch_x, None, None, None)
                        elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                            outputs = self.model(batch_x, self.args.pred_len, self.device)
                        elif self.args.model == 'Transformer':
                            encoder_input = batch_x
                            dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                            decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp],dim=1)
                            outputs = self.model(encoder_input, decoder_input, None, None)
                        elif self.args.model == 'CNN':
                            outputs = self.model(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model in ['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-xl']:
                        outputs = self.model(batch_x, None, None, None)
                    elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                        outputs = self.model(batch_x, self.args.pred_len, self.device)
                    elif self.args.model == 'Transformer':
                        encoder_input = batch_x
                        dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                        decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp],dim=1)
                        outputs = self.model(encoder_input, decoder_input, None, None)
                    elif self.args.model == 'CNN':
                        outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (iter_count + 1) % 100 == 0:
                    print("\t iters: {0}, epoch: {1} | loss: {2:.7f}".format(iter_count + 1, epoch + 1, loss.item()))
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} ,Cost time: {2:.4f} seconds | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                    epoch + 1, train_steps, time.time() - epoch_time, train_loss, vali_loss))
            
            visual_train_loss.append(train_loss)
            visual_vali_loss.append(vali_loss)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # TODO: adjust learning rate
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        # save loss 
        plt.plot(visual_train_loss, label="Train Loss", color='r')
        plt.plot(visual_vali_loss, label='Validation Loss', color='b')
        # plt.plot(visual_test_loss, label="Test Loss", color='g')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        mkdir(r"./lossJPG")
        
        plt.savefig(r"./lossJPG/{}".format(self.args.model))

        return self.model
    
    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-xl']:
                            outputs = self.model(batch_x, None, None, None)
                        elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                            outputs = self.model(batch_x, self.args.pred_len, self.device)
                        elif self.args.model == 'Transformer':
                            encoder_input = batch_x
                            dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                            decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp],dim=1)
                            outputs = self.model(encoder_input, decoder_input, None, None)
                        elif self.args.model == 'CNN':
                            outputs = self.model(batch_x)
                else:
                    if self.args.model in ['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-xl']:
                        outputs = self.model(batch_x, None, None, None)
                    elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                        outputs = self.model(batch_x, self.args.pred_len, self.device)
                    elif self.args.model == 'Transformer':
                        encoder_input = batch_x
                        dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                        decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp],dim=1)
                        outputs = self.model(encoder_input, decoder_input, None, None)
                    elif self.args.model == 'CNN':
                        outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, load = 0):
        test_data, test_loader = self._get_data(flag='test')

        print("loading model")
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting + self.args.save_name), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        
        flops = 0
        params = 0
        inference_times = 0
        preds = []
        trues = []
        inputx = []
        # test model
        inference_start = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-xl']:
                            outputs = self.model(batch_x, None, None, None)
                            flops1, params1 = profile(self.model, inputs=(batch_x, None, None, None), verbose=False)
                        elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                            outputs = self.model(batch_x, self.args.pred_len, self.device)
                            flops1, params1 = profile(self.model, inputs=(batch_x, self.args.pred_len, self.deveice), verbose=False)
                        elif self.args.model == 'Transformer':
                            encoder_input = batch_x
                            dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                            decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp],dim=1)
                            outputs = self.model(encoder_input, decoder_input, None, None)
                            flops1, params1 = profile(self.model, inputs=(encoder_input, decoder_input,), verbose=False)

                        elif self.args.model == 'CNN':
                            outputs = self.model(batch_x)
                            flops1, params1 = profile(self.model, inputs=(batch_x,), verbose=False)
                        inference_time = time.time() - inference_start
                else:
                    if self.args.model in ['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-xl']:
                        outputs = self.model(batch_x, None, None, None)
                        flops1, params1 = profile(self.model, inputs=(batch_x, None, None, None,), verbose=False)
                    elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                        outputs = self.model(batch_x, self.args.pred_len, self.device)
                        flops1, params1 = profile(self.model, inputs=(batch_x, self.args.pred_len, self.deveice,), verbose=False)
                    elif self.args.model == 'Transformer':
                        encoder_input = batch_x
                        dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                        decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp],dim=1)
                        outputs = self.model(encoder_input, decoder_input, None, None)
                        flops1, params1 = profile(self.model, inputs=(encoder_input, decoder_input,), verbose=False)
                    elif self.args.model == 'CNN':
                        outputs = self.model(batch_x)
                        flops1, params1 = profile(self.model, inputs=(batch_x,), verbose=False)
                    inference_time = time.time() - inference_start

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()[:, :, f_dim:]
                batch_y = batch_y.detach().cpu().numpy()[:, :, f_dim:]
                batch_x = batch_x.detach().cpu().numpy()[:, :, f_dim:]

                pred = outputs
                true = batch_y
                input = batch_x

                preds.append(pred)
                trues.append(true)
                inputx.append(input)

                flops += flops1
                params += params1
                inference_times += inference_time
        
        preds, trues, inputx = np.array(preds), np.array(trues), np.array(inputx)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape((-1,) + preds.shape[2:])
        trues = trues.reshape((-1,) + trues.shape[2:])
        inputx = inputx.reshape((-1,) + inputx.shape[2:])
        print(f"test shape transform to :{preds.shape} {trues.shape}")


        # get metrics
        mae, mse, rmse, mape, mspe, nmse = metric(preds[:, :, :64], trues[:, :, :64])
        flops = flops / (len(test_loader) * self.args.batch_size)
        params = params / (len(test_loader))
        inference_times = inference_times / (len(test_loader))

        # save results
        # npy
        directory_path = "./results_test/" + setting + "/"
        mkdir(directory_path)
        np.save(directory_path + "metrics.npy",np.array([mae, mse, rmse, mape, mspe, nmse, flops, params]),)
        np.save(directory_path + "pred.npy", preds)
        np.save(directory_path + "true.npy", trues)
        np.save(directory_path + "input.npy", inputx)
        
        # png
        self.visual(directory_path)

        return
    
    def visual(self, source_path):
        plt.clf()
        pred_path = os.path.join(source_path, "pred.npy")
        true_path = os.path.join(source_path, "true.npy")
        pred = np.load(pred_path, allow_pickle=True)
        true = np.load(true_path, allow_pickle=True)

        print("shape of pred:", pred.shape)
        print("shape of true:", true.shape)

        start_snapshot = 0
        end_snapshot = 200
        seq_num = 0 
        subcarrier_num = 0

        plt.figure(figsize=[15, 5])
        plt.plot(pred[start_snapshot:end_snapshot, seq_num, subcarrier_num], marker='o', markeredgecolor="black", linestyle="--", label='Pred', color = 'r',)
        plt.plot(true[start_snapshot:end_snapshot, seq_num, subcarrier_num], color='b', label='True')
        # plt.plot(pred[start_snapshot:end_snapshot, seq_num, subcarrier_num], color='y', label='Pred')
        plt.xlabel('Row Number')
        plt.ylabel('Value')
        plt.legend()
        plt.title(r'model{}_{}_snr{}_speed{}'.format(self.args.model, self.args.EmbeddingType, self.args.SNR, self.args.speed))
        plt.show()
        # plt.savefig('visualize.png'.format(source_path))
        plt.savefig('./visualize.png')
        
        # 绘制对比图
        plt.clf()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 矩阵1的热图
        axes[0].imshow(true[start_snapshot:end_snapshot, seq_num, :], cmap='viridis')
        axes[0].set_title("Trues")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        # 矩阵2的热图
        axes[1].imshow(pred[start_snapshot:end_snapshot, seq_num, :], cmap='plasma')
        axes[1].set_title("Preds")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        plt.savefig('{}Heatmap.png'.format(source_path))

        return 
