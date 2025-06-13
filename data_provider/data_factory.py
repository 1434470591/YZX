from data_provider.data_loader import Dataset_Pro, SJTU
from torch.utils.data import DataLoader

data_dict = {
    'QuaDriGa':Dataset_Pro,
    'CHINA':Dataset_Pro,
    'SJTU':SJTU,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    if args.data == 'QuaDriGa':
        if flag != 'test' and flag != 'TEST':
            data_set = Data(file_path_r=args.root_path + '/Training Dataset/H_U_his_train.mat',
                                file_path_t=args.root_path + '/Training Dataset/H_D_pre_train.mat' if args.is_U2D else  args.root_path + '/Training Dataset/H_U_pre_train.mat',
                                is_train=1 if flag == 'train' or flag == 'TRAIN' else 0,
                                #    ir=args.ir,
                                #    SNR=args.SNR,
                                is_U2D=args.is_U2D,
                                is_few=args.is_few,
                                train_per=args.train_per,
                                valid_per=args.valid_per)
    elif args.data == 'SJTU':
        data_set = Data(args, flag)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.drop_last)
    
    return data_set, data_loader
