from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_100_100, Dataset_Financial_Info,\
    Dataset_Financial_LLM_Train, Dataset_Solar_Info, Dataset_Solar_LLM_Train
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    '100_100': Dataset_100_100,
    'financial_info': Dataset_Financial_Info,
    'financial_llm_train': Dataset_Financial_LLM_Train,
    'solar_info': Dataset_Solar_Info,
    'solar_llm_train': Dataset_Solar_LLM_Train
}

def custom_collate_fn(batch):
    # batch is a list of tuples: (seq_x, seq_x_semantic, seq_x_time, seq_y, seq_y_semantic, seq_y_time)
    seq_x, seq_x_sem, seq_x_time, seq_y, seq_y_sem, seq_y_time = zip(*batch)

    # Convert numpy arrays to tensors before stacking
    seq_x = torch.stack([torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in seq_x])
    seq_x_sem = torch.stack([torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in seq_x_sem])
    seq_y = torch.stack([torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in seq_y])
    seq_y_sem = torch.stack([torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in seq_y_sem])

    # Keep time strings as list (no stacking)
    return seq_x, seq_x_sem, list(seq_x_time), seq_y, seq_y_sem, list(seq_y_time)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        
        if args.data == 'custom':
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=custom_collate_fn
            )    
        
        return data_set, data_loader
