import argparse
import os
import torch
import torch.backends
import random
import numpy as np
from utils.print_args import print_args
from exp.exp_QuaDriGa import Exp_QuaDriGa
from exp.exp_SJTU import Exp_SJTU
if __name__ == '__main__':
    seed = 42
    random.seed(seed)                  # Python 自带随机模块
    np.random.seed(seed)               # NumPy 随机模块
    torch.manual_seed(seed)            # CPU 上的 torch 随机性
    torch.cuda.manual_seed(seed)       # GPU 上的 torch 随机性（单卡）
    torch.cuda.manual_seed_all(seed)   # 多卡情况下所有 GPU 的种子
    torch.backends.cudnn.deterministic = True  # 避免非确定性算法
    torch.backends.cudnn.benchmark = False     # 禁止自动优化算法（可能引入不确定性）

    parser = argparse.ArgumentParser(description='LLM4CP')
    # basic config
    parser.add_argument('--task_name', type=str, default='Forecasting',
                        help='task name, options:[Forecasting, Classification]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='LLMs',
                        help='model name, options: [LLMs, Transformer, CNN, RNN, LSTM, GRU]')

    # data loader
    parser.add_argument('--data', type=str, default='QuaDriGa', help='dataset type, options: [QuaDriGa, CHINA, SJTU]')
    parser.add_argument('--root_path', type=str, default='/workspace/DATASETs/', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # SJTU dataset
    parser.add_argument('--data_path', type=str, default='IQ_CFR_SNR100.csv', help='data file')
    parser.add_argument('--SNR_ID', type=int, default=100, help='SNR ID in SJTU dataset')
    parser.add_argument("--scale", type=bool, default=True, help="standardization stats")
    parser.add_argument("--subcarrier_num", type=int, default=32, help="number of Subcarriers")
    parser.add_argument('--slid_step', type=int, default=1, help='sliding step of data loader')
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[0.8, 0.1, 0.1],
                        help='split ratio of train, validation and test set')
    parser.add_argument('--enhancement', type=bool, default=False, help='whether to use enhancement')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    # QuaDriGa dataset
    parser.add_argument('--is_U2D', type=int, default=0, help='whether to use U2D learning')
    parser.add_argument('--is_few', type=int, default=0, help='whether to use few-shot learning')
    parser.add_argument('--train_per', type=float, default=0.9, help='train set proportion')
    parser.add_argument('--valid_per', type=float, default=0.1, help='validation set proportion')

    # forecasting task
    parser.add_argument('--prev_len', type=int, default=16, help='previous sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')

    # model define
    # transformer
    parser.add_argument('--stack', type=int, default=1, help='stack Transformer')
    parser.add_argument('--enc_in', type=int, default=96, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=96, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=96, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=6, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--dropout_trans', type=float, default=0.2, help='dropout of transformer')
    parser.add_argument('--attn', type=str, default='full',
                        help='attention type, options: [prob, full]')  # prob, full
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder', default=False)
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=False)
    # rnn
    parser.add_argument('--output_size', type=int, default=96, help='output size of rnn')
    parser.add_argument('--input_size', type=int, default=96, help='input size of rnn')
    parser.add_argument('--hidden_size', type=int, default=192, help='hidden size of rnn')
    parser.add_argument('--num_layers', type=int, default=4, help='number of rnn layers')
    parser.add_argument('--dropout_rnn', type=float, default=0.1, help='dropout of rnn')
    
    # gpt2
    parser.add_argument('--llm_type', type=str, default='gpt2', help='llm type, options: [none, gpt2, gpt2-medium, gpt2-large, gpt2-x, llama]')
    parser.add_argument('--llm_layers', type=int, default=6, help='number of gpt2 layers')
    parser.add_argument('--llm_d_model', type=int, default=768, help='dimension of model in gpt2')
    parser.add_argument('--llm_d_ff', type=int, default=768, help='dimension of ffn in gpt2')
    parser.add_argument('--mlp', type=int, default=0, help='whether to use mlp in gpt2')
    parser.add_argument('--res_layers', type=int, default=4, help='number of res layers in gpt2')
    parser.add_argument('--K', type=int, default=48, help='number of channels in gpt2')
    parser.add_argument('--UQh', type=int, default=1, help='number of UQh in gpt2')
    parser.add_argument('--UQv', type=int, default=1, help='number of UQv in gpt2')
    parser.add_argument('--BQh', type=int, default=1, help='number of BQh in gpt2')
    parser.add_argument('--BQv', type=int, default=1, help='number of BQv in gpt2')
    parser.add_argument('--patch_size', type=int, default=4, help='patch size in gpt2')
    parser.add_argument('--stride', type=int, default=1, help='stride in gpt2')
    parser.add_argument('--res_dim', type=int, default=64, help='res dimension in gpt2')
    parser.add_argument('--dropout_gpt', type=float, default=0.1, help='dropout in gpt2')
    parser.add_argument('--pretrain', type=int, default=1, help='Use pre-trained LLMs')
    parser.add_argument('--freeze', type=int, default=1, help='Freeze GPT2 parameters')
    parser.add_argument('--lora', type=int, default=1, help='whether to use lora in gpt2')
    parser.add_argument('--description', type=str, default='', help='description of datasets')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--drop_last', type=bool, default=True, help='drop last batch in data loader')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='description of experiment')
    parser.add_argument('--loss_func', type=str, default='NMSE', help='loss function, options: [MSE, NMSE]')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate.options: [cosine, constant]')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--speculative', type=int, default=0, help='whether to use speculative testing, 0: no, 1: yes')
    parser.add_argument('--freq', type=int, default=3, help='frequency of speculative testing, only works when speculative is 1')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()

    if args.data == 'STJU': 
        args.batch_size = 1024

    if args.model == 'LLMs' and args.llm_type == 'llama':
        args.use_multi_gpu = True

    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print(f'Using GPU: {args.gpu}')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.data == 'QuaDriGa':
        Exp = Exp_QuaDriGa
    elif args.data == 'SJTU':
        Exp = Exp_SJTU
        args.data_path = f'IQ_CFR_SNR{args.SNR_ID}.csv'
        args.batch_size = 16  # SJTU dataset is small, so we can use a larger batch size

    if args.is_training:
        args.model_id = 'train'
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
                args.task_name,
                args.data,
                args.model,
                args.llm_type,
                args.features,
                args.prev_len,
                args.label_len,
                args.pred_len,
                args.des, 
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            print('>>>>>>>inference : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.pred(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        args.model_id = 'test'
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
            args.task_name,
            args.data,
            args.model,
            args.llm_type,
            args.features,
            args.prev_len,
            args.label_len,
            args.pred_len,
            args.des, 
            ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
