from data_provider.graph_data_loader import InductivePEMS
from torch.utils.data import DataLoader


def data_provider(args, flag):
    dataset = InductivePEMS
    if flag != 'train':
        shuffle_flag = False
        drop_last = False
        batch_size = 256
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = dataset(data_path=args.data_path,
                       size=[args.input_len, args.look_back, args.pred_len],
                       flag=flag,
                       scale=not args.no_scale,
                       freq=freq,
                       inverse=args.inverse,
                       target_features=args.target_features,
                       input_features=args.input_features,
                       dloader_name=args.dloader_name,
                       step=args.slide_step,
                       splits=[args.train_ratio, args.valid_ratio],
                       unknown_nodes_path=args.unknown_nodes_path,
                       unknown_nodes_num=args.unknown_nodes_num,
                       sp_adj=args.sp_adj)

    return DataLoader(data_set,
                      batch_size=batch_size,
                      shuffle=shuffle_flag,
                      num_workers=args.num_workers,
                      drop_last=drop_last)
