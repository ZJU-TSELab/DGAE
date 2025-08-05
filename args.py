import argparse


def initialize_args():
    parser = argparse.ArgumentParser(description='DoubleSequece arguments')

    parser.add_argument('--transfer_only', action='store_true', help='transfer only')
    parser.add_argument('--pretrained_model_path', type=str, help='pretrained model path')
    # train
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--loss',
                        type=str,
                        default='mse',
                        help='loss function used for backpropagation, mse, mae...')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to train on')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adamw',
                        choices=['adam', 'sgd', 'adamw'],
                        help='optimizer')
    parser.add_argument('--scheduler',
                        type=str,
                        default='plateau',
                        choices=['plateau', 'step', 'cosine', 'None', 'multistep'],
                        help='scheduler:plateau, step, cosine, None,...')
    parser.add_argument("--scheduler_patience", default=1, help="scheduler patience")
    parser.add_argument('--patience', type=int, default=3, help='early stopping epochs')
    parser.add_argument('--stop_based',
                        type=str,
                        default='train_mask',
                        choices=['val_total', 'val_mask', 'train_total', 'train_mask'],
                        help='early stopping criteria')
    parser.add_argument('--return_best', action='store_true', help='return best model')
    parser.add_argument('--test_only', action='store_true', help='test only')
    parser.add_argument('--read_only', action='store_true', help='read only')

    parser.add_argument('--t_mark', action='store_true', help='add time stamp to feature')
    parser.add_argument('--pos_mark',
                        action='store_true',
                        help='add pos stamp to feature')
    parser.add_argument('--miss_mark',
                        action='store_true',
                        help='add miss index to feature')
    # save and logging settings
    parser.add_argument('--best_checkpoint_path',
                        type=str,
                        default='./check_points/',
                        help='best val checkpoint path')
    parser.add_argument('--best_prediction_path',
                        type=str,
                        default='./predictions/',
                        help='results path')
    parser.add_argument('--best_metrics_path',
                        type=str,
                        default='./',
                        help='results path')
    parser.add_argument('--best_fig_path',
                        type=str,
                        default='./figures/',
                        help='figures path')
    parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--wandb_entity',
                        type=str,
                        default='zqslalala',
                        help='user name of wandb')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='log path')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--log_interval', type=int, default=100)

    # Model
    parser.add_argument('--model', type=str)
    parser.add_argument('--latent_dim', type=int, default=32, help='latent dim of vae')
    parser.add_argument('--k_hop', type=int, default=1, help='k_hop for graph')
    parser.add_argument('--diff_alpha',
                        type=float,
                        default=1.,
                        help='alpha for diffusion kernel')
    #feature propagation
    parser.add_argument('--fp_step',
                        type=int,
                        default=-1,
                        help='feature propagation step')
    parser.add_argument('--sp_adj', action='store_true', help='use spatial adj')

    # task
    parser.add_argument('--input_len', type=int, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--look_back', type=int, default=0)
    parser.add_argument('--slide_step',
                        type=int,
                        default=1,
                        help='step for sliding window')
    parser.add_argument('--target_features',
                        type=str,
                        required=True,
                        choices=['q', 'o', 'v', 'qv', 'qov'],
                        help='q is flow, v is speed, o is occupancy')
    parser.add_argument('--input_features',
                        type=str,
                        required=True,
                        choices=['q', 'o', 'v', 'qv', 'qov'],
                        help='q is flow, v is speed, o is occupancy')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='train ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='validation ratio')
    parser.add_argument(
        '--task',
        type=str,
        default='kriging',
        help='task type: predict, kriging, predict_kriging, kriging_predict')
    #注意这里人工模拟缺失，是为了最终测试。而训练阶段还需要对输入接着抽出一部分缺失，然后并利用这部分的label来计算loss
    #三种方式模拟缺失，使用优先度同定义顺序
    parser.add_argument('--unknown_nodes_path',
                        type=str,
                        help='path for unobserved sensors id in the train data')
    parser.add_argument('--unknown_nodes_num',
                        type=int,
                        default=0,
                        help='number of unobserved sensors  in the training data')

    parser.add_argument('--save_model',
                        action='store_true',
                        help='save model not model state dict')
    # Dataset
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dloader_name', type=str, default='PEMS')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--no_scale', action='store_true')
    parser.add_argument('--inverse', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--freq', type=str, default='min')

    args = parser.parse_args()

    assert args.input_features in args.target_features, 'input_features should be in target_features'
    args.c_out = len(args.target_features)
    args.enc_in = len(args.input_features)
    print(f'enc_in and c_out is set automatically', )
    if args.scheduler == 'multistep':
        args.scheduler_patience = [int(x) for x in args.scheduler_patience.split(',')]
    else:
        args.scheduler_patience = int(args.scheduler_patience)
    return args
