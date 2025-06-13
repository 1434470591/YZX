from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.losses import SE_Loss, NMSELoss
from utils.pad import PAD3
from utils.prev import pronyvec
import torch
import torch.nn as nn
from torch import optim
import os
import csv
import time
import warnings
import numpy as np
import hdf5storage
from einops import rearrange
from data_provider.data_loader import LoadBatch_ofdm_1, LoadBatch_ofdm_2, noise, Transform_TDD_FDD
warnings.filterwarnings('ignore')


class Exp_QuaDriGa(Exp_Basic):
    def __init__(self, args):
        super(Exp_QuaDriGa, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_gpu and self.args.use_multi_gpu and self.args.model_id == 'train':
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=self.args.device_ids)#这一行算不算分片
        else :
            model = model.to(self.device)

        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            print('Use Data DP')

        for name, param in model.named_parameters():
            if param.device == torch.device('cpu'):
                print(f"Parameter {name} is on {param.device}")
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))

        total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim

    def _select_criterion(self):
        if self.args.loss_func == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss_func == 'NMSE':
            criterion = NMSELoss()
        return criterion
 

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['LLMs']:
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
                    if self.args.model in ['LLMs']:
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

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("Train Steps", train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion().to(self.device)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['LLMs']:
                            print("inputs_embeds device:", batch_x.device)
                            print("model.device:", next(self.model.parameters()).device)
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
                    if self.args.model in ['LLMs']:
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

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {} s".format(epoch + 1, time.time() - epoch_time))
            # train_loss = np.average(train_loss)
            train_loss = np.nanmean(np.array(train_loss))
            vali_loss = self.vali(vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss,))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # if self.args.model in ['RNN', 'LSTM', 'GRU']:
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def test(self, setting):
        draft_model = self.model_dict['CNN'].Model(self.args).float().to(self.device)

        if self.args.model not in ['pad', 'pvec', 'np']:
            # seq_len = prev_len + pred_len = 20
            print('loading model')
            if self.args.use_gpu and self.args.use_multi_gpu and self.args.model_id == 'train':
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
            # 单卡推理
            else :
                state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device)
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)

                state_dict_draft = torch.load(os.path.join('/workspace/YZX/checkpoints/Forecasting_QuaDriGa_CNN_gpt2_ftM_sl16_ll12_pl4_Exp_0/checkpoint.pth'), map_location=self.device)
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict_draft.items()}
                draft_model.load_state_dict(new_state_dict)

        print('loading data')

        prev_path = self.args.root_path + "Testing Dataset/H_U_his_test.mat"  # path of dataset [H_D_his_test]
        pred_path_tdd = self.args.root_path + "Testing Dataset/H_U_pre_test.mat"  # path of dataset [H_D_pre_test]
        pred_path_fdd = self.args.root_path + "Testing Dataset/H_D_pre_test.mat"  # path of dataset [H_U_pre_test]

        test_data_prev_base = hdf5storage.loadmat(prev_path)['H_U_his_test']
        test_data_pred_base = hdf5storage.loadmat(pred_path_fdd)['H_D_pre_test'] if self.args.is_U2D else hdf5storage.loadmat(pred_path_tdd)['H_U_pre_test']
        print("---------------------------------------------------------------")
        
        preds = []
        trues = []
        INFERENCE_TIME = []
        NMSE = []
        SE = []
        K, Nt, Nr, SR = (48, 16, 1, 1)
        # load model and test
        criterion = NMSELoss()
        criterion_se = SE_Loss(snr=10, device=self.device)

        for speed in range(0, 10):
            test_loss_stack = []
            test_loss_stack_se = []
            test_loss_stack_se0 = []
            inference_time = []
            test_data_prev = test_data_prev_base[[speed], ...]
            test_data_pred = test_data_pred_base[[speed], ...]
            test_data_prev = rearrange(test_data_prev, 'v b l k n m c -> (v b c) (n m) l (k)')
            test_data_pred = rearrange(test_data_pred, 'v b l k n m c -> (v b c) (n m) l (k)')
            test_data_prev = noise(test_data_prev, 18)
            test_data_pred = noise(test_data_pred, 18)
            std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
            test_data_prev = test_data_prev / std
            test_data_pred = test_data_pred / std
            lens, _, _, _ = test_data_prev.shape
            if self.args.model in ['LLMs', 'Transformer', 'RNN', 'LSTM', 'GRU', 'CNN', 'NP']:
                if self.args.model != 'NP':
                    self.model.eval()
                prev_data = LoadBatch_ofdm_2(test_data_prev)
                pred_data = LoadBatch_ofdm_2(test_data_pred)
                bs = 64
                cycle_times = lens // bs
                with torch.no_grad():
                    for cyt in range(cycle_times):
                        prev = prev_data[cyt * bs:(cyt + 1) * bs, :, :].to(self.device)
                        pred = pred_data[cyt * bs:(cyt + 1) * bs, :, :].to(self.device)
                        prev = rearrange(prev, 'b m l k -> (b m) l k')
                        pred = rearrange(pred, 'b m l k -> (b m) l k')
                        if self.args.speculative == 0:
                            if self.args.model in ['LLMs']:
                                start_time = time.time()
                                out = self.model(prev, None, None, None)
                                cost_time = time.time() - start_time

                            elif self.args.model == 'Transformer':
                                start_time = time.time()
                                encoder_input = prev
                                dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                                decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp], dim=1)
                                out = self.model(encoder_input, decoder_input)
                                cost_time = time.time() - start_time

                            elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                                start_time = time.time()
                                out = self.model(prev, self.args.pred_len, self.device)
                                cost_time = time.time() - start_time

                            elif self.args.model == 'CNN':
                                start_time = time.time()
                                out = self.model(prev)
                                cost_time = time.time() - start_time

                            elif self.args.model == 'NP':
                                out = prev[:, [-1], :].repeat([1, self.args.pred_len, 1])
                        else:
                            if cyt % self.args.freq == 0:
                                if self.args.model in ['LLMs']:
                                    start_time = time.time()
                                    out = self.model(prev, None, None, None)
                                    cost_time = time.time() - start_time
                                elif self.args.model == 'Transformer':
                                    start_time = time.time()
                                    encoder_input = prev
                                    dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                                    decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp], dim=1)
                                    out = self.model(encoder_input, decoder_input)
                                    cost_time = time.time() - start_time

                                elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                                    start_time = time.time()
                                    out = self.model(prev, self.args.pred_len, self.device)
                                    cost_time = time.time() - start_time

                            else:
                                start_time = time.time()
                                out = draft_model(prev)
                                cost_time = time.time() - start_time

                        loss = criterion(out, pred)
                        out = rearrange(out, '(b m) l k -> b l (k m)', b=bs)
                        pred = rearrange(pred, '(b m) l k -> b l (k m)', b=bs)

                        se, se0 = criterion_se(h=Transform_TDD_FDD(out, Nt=4*4, Nr=1),
                                            h0=Transform_TDD_FDD(pred, Nt=4*4, Nr=1))
                        inference_time.append(cost_time)
                        test_loss_stack.append(loss.item())
                        test_loss_stack_se.append(se.item())
                        test_loss_stack_se0.append(se0.item())

                        preds.append(out.detach().cpu().numpy())    
                        trues.append(pred.detach().cpu().numpy())
                print( "speed: ", speed, "inference_time: ", np.mean(np.array(inference_time)), "NMSE: ", np.nanmean(np.array(test_loss_stack)),
                    "SE: ", -np.nanmean(np.array(test_loss_stack_se)), "SE0: ", -np.nanmean(np.array(test_loss_stack_se0)),
                    "SE_per ", np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
                INFERENCE_TIME.append(np.mean(np.array(inference_time)))
                NMSE.append(np.nanmean(np.array(test_loss_stack)))
                SE.append(np.nanmean(np.array(test_loss_stack_se)))
                # SE.append(np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
            elif self.args.model in ['pad', 'pvec']:
                cycle_times = lens
                for cyt in range(cycle_times):
                    prev = test_data_prev[cyt, :, :, :]
                    prev = rearrange(prev, 'm l k -> k l m', k=K)
                    pred = test_data_pred[cyt, :, :, :]
                    pred = rearrange(pred, 'm l k -> k l m', k=K)
                    start_time = time.time()
                    if self.args.model == 'pad':
                        # outputs_AR_delay
                        out = PAD3(prev, p=8, startidx=self.args.prev_len, subcarriernum=K, Nr=Nr, Nt=Nt,
                                pre_len=self.args.pred_len)
                    elif self.args.model == 'pvec':
                        # outputs_AR_freq
                        out = pronyvec(prev, p=8, startidx=self.args.prev_len, subcarriernum=K, Nr=Nr, Nt=Nt,
                                    pre_len=self.args.pred_len)
                    cost_time = time.time() - start_time
                    out = LoadBatch_ofdm_1(out)
                    pred = LoadBatch_ofdm_1(pred)
                    loss = criterion(out, pred)
                    se, se0 = criterion_se(h=Transform_TDD_FDD(out, Nt=4*4, Nr=1), h0=Transform_TDD_FDD(pred, Nt=4*4, Nr=1))
                    inference_time.append(cost_time)
                    test_loss_stack.append(loss.item())
                    test_loss_stack_se.append(se.item())
                    test_loss_stack_se0.append(se0.item())
                    preds.append(out.detach().cpu().numpy())    
                    trues.append(pred.detach().cpu().numpy())

                print("speed: ", speed, "inference_time: ", np.mean(np.array(inference_time)), "NMSE: ", np.nanmean(np.array(test_loss_stack)), 
                    "SE: ", -np.nanmean(np.array(test_loss_stack_se)), "SE0: ", -np.nanmean(np.array(test_loss_stack_se0)),
                    "SE_per ", np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
                INFERENCE_TIME.append(np.mean(np.array(inference_time)))
                NMSE.append(np.nanmean(np.array(test_loss_stack)))
                SE.append(np.nanmean(np.array(test_loss_stack_se)))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, nmse = metric(preds, trues)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, nmse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        row = [[self.args.model]] if self.args.model != 'LLMs' else [[self.args.llm_type]]
        # for speed, nmse, se, it in zip(range(10), NMSE, SE, INFERENCE_TIME):
        #     row.append([speed, nmse, se, it])
        if self.args.model in ['Transformer', ]:
            row.append([self.args.e_layers, self.args.d_layers, 10 * np.log10(nmse), nmse, np.nanmean(NMSE), np.nanmean(SE), np.nanmean(INFERENCE_TIME)])
        elif self.args.model in ['LLMs', ]:
            row.append([self.args.lradj, 10 * np.log10(nmse), nmse, np.nanmean(NMSE), np.nanmean(SE), np.nanmean(INFERENCE_TIME)])
        else:
            row.append([10 * np.log10(nmse), nmse, np.nanmean(NMSE), np.nanmean(SE), np.nanmean(INFERENCE_TIME)])
        row.append([])
        with open('./output_csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(row)  # 多行写入
        return
    
    def pred(self, setting):
        draft_model = self.model_dict['CNN'].Model(self.args).float().to(self.device)

        if self.args.model not in ['pad', 'pvec', 'np']:
            # seq_len = prev_len + pred_len = 20
            print('loading model')
            if self.args.use_gpu and self.args.use_multi_gpu and self.args.model_id == 'train':
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
            # 单卡推理
            else :
                state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device)
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)

                state_dict_draft = torch.load(os.path.join('/workspace/YZX/checkpoints/Forecasting_QuaDriGa_CNN_gpt2_ftM_sl16_ll12_pl4_Exp_0/checkpoint.pth'), map_location=self.device)
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict_draft.items()}
                draft_model.load_state_dict(new_state_dict)

        print('loading data')

        prev_path = self.args.root_path + "Testing Dataset/H_U_his_test.mat"  # path of dataset [H_D_his_test]
        pred_path_tdd = self.args.root_path + "Testing Dataset/H_U_pre_test.mat"  # path of dataset [H_D_pre_test]
        pred_path_fdd = self.args.root_path + "Testing Dataset/H_D_pre_test.mat"  # path of dataset [H_U_pre_test]

        test_data_prev_base = hdf5storage.loadmat(prev_path)['H_U_his_test']
        test_data_pred_base = hdf5storage.loadmat(pred_path_fdd)['H_D_pre_test'] if self.args.is_U2D else hdf5storage.loadmat(pred_path_tdd)['H_U_pre_test']
        print("---------------------------------------------------------------")
        
        preds = []
        trues = []
        INFERENCE_TIME = []
        NMSE = []
        SE = []
        K, Nt, Nr, SR = (48, 16, 1, 1)
        # load model and test
        criterion = NMSELoss()
        criterion_se = SE_Loss(snr=10, device=self.device)

        for speed in range(0, 10):
            test_loss_stack = []
            test_loss_stack_se = []
            test_loss_stack_se0 = []
            inference_time = []

            test_data_prev = test_data_prev_base[[speed], ...]
            test_data_pred = test_data_pred_base[[speed], ...]
            test_data_prev = rearrange(test_data_prev, 'v b l k n m c -> (v b c) (n m) l (k)')
            test_data_pred = rearrange(test_data_pred, 'v b l k n m c -> (v b c) (n m) l (k)')
            test_data_prev = noise(test_data_prev, 18)
            test_data_pred = noise(test_data_pred, 18)

            std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
            test_data_prev = test_data_prev / std
            test_data_pred = test_data_pred / std

            lens, _, _, _ = test_data_prev.shape
            if self.args.model in ['LLMs', 'Transformer', 'RNN', 'LSTM', 'GRU', 'CNN', 'NP']:
                if self.args.model != 'NP':
                    self.model.eval()
                prev_data = LoadBatch_ofdm_2(test_data_prev)
                pred_data = LoadBatch_ofdm_2(test_data_pred)
                bs = 1
                cycle_times = lens // bs
                with torch.no_grad():
                    for cyt in range(cycle_times):
                        prev = prev_data[cyt * bs:(cyt + 1) * bs, :, :].to(self.device)
                        pred = pred_data[cyt * bs:(cyt + 1) * bs, :, :].to(self.device)
                        
                        prev = rearrange(prev, 'b m l k -> (b m) l k')
                        pred = rearrange(pred, 'b m l k -> (b m) l k')
                        if self.args.speculative == 0:
                            if self.args.model in ['LLMs']:
                                start_time = time.time()
                                out = self.model(prev, None, None, None)
                                cost_time = time.time() - start_time

                            elif self.args.model == 'Transformer':
                                start_time = time.time()
                                encoder_input = prev
                                dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                                decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp], dim=1)
                                out = self.model(encoder_input, decoder_input)
                                cost_time = time.time() - start_time

                            elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                                start_time = time.time()
                                out = self.model(prev, self.args.pred_len, self.device)
                                cost_time = time.time() - start_time

                            elif self.args.model == 'CNN':
                                start_time = time.time()
                                out = self.model(prev)
                                cost_time = time.time() - start_time

                            elif self.args.model == 'NP':
                                out = prev[:, [-1], :].repeat([1, self.args.pred_len, 1])
                        else:
                            if cyt % self.args.freq:
                                start_time = time.time()
                                out = draft_model(prev)
                                cost_time = time.time() - start_time
                            else:
                                if self.args.model in ['LLMs']:
                                    start_time = time.time()
                                    out = self.model(prev, None, None, None)
                                    cost_time = time.time() - start_time
                                elif self.args.model == 'Transformer':
                                    start_time = time.time()
                                    encoder_input = prev
                                    dec_inp = torch.zeros_like(encoder_input[:, -self.args.pred_len:, :]).to(self.device)
                                    decoder_input = torch.cat([encoder_input[:, self.args.prev_len - self.args.label_len:self.args.prev_len, :], dec_inp], dim=1)
                                    out = self.model(encoder_input, decoder_input)
                                    cost_time = time.time() - start_time

                                elif self.args.model in ['RNN', 'LSTM', 'GRU']:
                                    start_time = time.time()
                                    out = self.model(prev, self.args.pred_len, self.device)
                                    cost_time = time.time() - start_time

                        loss = criterion(out, pred)
                        out = rearrange(out, '(b m) l k -> b l (k m)', b=bs)
                        pred = rearrange(pred, '(b m) l k -> b l (k m)', b=bs)

                        se, se0 = criterion_se(h=Transform_TDD_FDD(out, Nt=4*4, Nr=1),
                                            h0=Transform_TDD_FDD(pred, Nt=4*4, Nr=1))
                        inference_time.append(cost_time)
                        test_loss_stack.append(loss.item())
                        test_loss_stack_se.append(se.item())
                        test_loss_stack_se0.append(se0.item())

                        preds.append(out.detach().cpu().numpy())    
                        trues.append(pred.detach().cpu().numpy())
                print( "speed: ", speed, "inference_time: ", np.mean(np.array(inference_time)), "NMSE: ", np.nanmean(np.array(test_loss_stack)),
                    "SE: ", -np.nanmean(np.array(test_loss_stack_se)), "SE0: ", -np.nanmean(np.array(test_loss_stack_se0)),
                    "SE_per ", np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
                INFERENCE_TIME.append(np.mean(np.array(inference_time)))
                NMSE.append(np.nanmean(np.array(test_loss_stack)))
                SE.append(np.nanmean(np.array(test_loss_stack_se)))
                # SE.append(np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
            elif self.args.model in ['pad', 'pvec']:
                cycle_times = lens
                for cyt in range(cycle_times):
                    prev = test_data_prev[cyt, :, :, :]
                    prev = rearrange(prev, 'm l k -> k l m', k=K)
                    pred = test_data_pred[cyt, :, :, :]
                    pred = rearrange(pred, 'm l k -> k l m', k=K)
                    start_time = time.time()
                    if self.args.model == 'pad':
                        # outputs_AR_delay
                        out = PAD3(prev, p=8, startidx=self.args.prev_len, subcarriernum=K, Nr=Nr, Nt=Nt,
                                pre_len=self.args.pred_len)
                    elif self.args.model == 'pvec':
                        # outputs_AR_freq
                        out = pronyvec(prev, p=8, startidx=self.args.prev_len, subcarriernum=K, Nr=Nr, Nt=Nt,
                                    pre_len=self.args.pred_len)
                    cost_time = time.time() - start_time
                    out = LoadBatch_ofdm_1(out)
                    pred = LoadBatch_ofdm_1(pred)
                    loss = criterion(out, pred)
                    se, se0 = criterion_se(h=Transform_TDD_FDD(out, Nt=4*4, Nr=1), h0=Transform_TDD_FDD(pred, Nt=4*4, Nr=1))
                    inference_time.append(cost_time)
                    test_loss_stack.append(loss.item())
                    test_loss_stack_se.append(se.item())
                    test_loss_stack_se0.append(se0.item())
                    preds.append(out.detach().cpu().numpy())    
                    trues.append(pred.detach().cpu().numpy())

                print("speed: ", speed, "inference_time: ", np.mean(np.array(inference_time)), "NMSE: ", np.nanmean(np.array(test_loss_stack)), 
                    "SE: ", -np.nanmean(np.array(test_loss_stack_se)), "SE0: ", -np.nanmean(np.array(test_loss_stack_se0)),
                    "SE_per ", np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
                INFERENCE_TIME.append(np.mean(np.array(inference_time)))
                NMSE.append(np.nanmean(np.array(test_loss_stack)))
                SE.append(np.nanmean(np.array(test_loss_stack_se)))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, nmse = metric(preds, trues)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, nmse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        row = [[self.args.model]] if self.args.model != 'LLMs' else [[self.args.llm_type]]
        # for speed, nmse, se, it in zip(range(10), NMSE, SE, INFERENCE_TIME):
        #     row.append([speed, nmse, se, it])
        if self.args.model in ['Transformer', ]:
            row.append([self.args.e_layers, self.args.d_layers, 10 * np.log10(nmse), nmse, np.nanmean(NMSE), np.nanmean(SE), np.nanmean(INFERENCE_TIME)])
        elif self.args.model in ['LLMs', ]:
            row.append([self.args.lradj, 10 * np.log10(nmse), nmse, np.nanmean(NMSE), np.nanmean(SE), np.nanmean(INFERENCE_TIME)])
        else:
            row.append([10 * np.log10(nmse), nmse, np.nanmean(NMSE), np.nanmean(SE), np.nanmean(INFERENCE_TIME)])
        row.append([])
        with open('./output_csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(row)  # 多行写入
        return