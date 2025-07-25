"""
Training functions and helpers
"""
import torch
import pickle
import torch
from .test import read_out
import os


def inference_model(model, data_loader, args, creterions):
    total_y_true = []
    total_y_pred = []
    total_y_true_trans = []
    total_y_pred_trans = []
    args.best_total_y = {}  # save best epoch test results
    print('-> Start inference...')
    inverse_transform = data_loader.dataset.inverse_transform
    model.eval()
    with torch.no_grad():
        for batch_ix, batch in enumerate(data_loader):
            print(f'Inference batch {batch_ix + 1}/{len(data_loader)}', end='\r')
            batch = [x.float().to(args.device) for x in batch]
            x_enc, x_dec, enc_t_mark, dec_t_mark = batch
            if not args.t_mark:
                enc_t_mark = None
            y = x_dec[:, -args.pred_len:].clone()
            x_dec = x_dec[:, -args.pred_len:]
            adj, pos_mark = data_loader.dataset.get_adj_and_pos_mark()
            if not args.pos_mark:
                pos_mark = None
            adj = torch.from_numpy(adj).float().to(args.device)
            if pos_mark is not None:
                pos_mark = torch.from_numpy(pos_mark).float().to(args.device)
            x_enc[torch.isnan(x_enc)] = 0  #在源数据集中，有些异常值被标注为0，这些异常值不参加任何评价计算
            x_dec = None
            if args.model == 'IGNNK':
                y_hat = model(adj, x_enc, enc_t_mark, pos_mark, x_dec, dec_t_mark)
            else:
                _, batch_mask, _, _ = data_loader.dataset.get_know_unknow_nodes()
                if args.miss_mark:
                    miss_mark = (x_enc == 0).float()
                else:
                    miss_mark = None
                y_hat, *_ = model(adj, x_enc, batch_mask, enc_t_mark, miss_mark, pos_mark)
            y = y.detach().cpu()
            y_hat = y_hat.detach().cpu()
            inverse_y = inverse_transform(y)
            inverse_y_hat = inverse_transform(y_hat)
            total_y_true.append(y)
            total_y_pred.append(y_hat)
            total_y_true_trans.append(inverse_y)
            total_y_pred_trans.append(inverse_y_hat)
        total_y = {
            'true': torch.cat(total_y_true, dim=0),
            'pred': torch.cat(total_y_pred, dim=0),
            'true_trans': torch.cat(total_y_true_trans, dim=0),
            'pred_trans': torch.cat(total_y_pred_trans, dim=0),
        }
        args.best_total_y['test'] = total_y
        print("-> save predictions and metrics...")
        if args.filling:
            args.best_prediction_path = f'./kriging_data/{args.model}/'
        else:
            args.best_prediction_path = args.best_prediction_path + args.exp_id + '/'
        print(f'save to {args.best_prediction_path}')
        if not os.path.exists(args.best_prediction_path):
            os.makedirs(args.best_prediction_path)
        with open(args.best_prediction_path + 'inference_total_y.pkl', 'wb') as f:
            args.best_total_y['known_nodes'], args.best_total_y[
                'unknown_nodes'] = data_loader.dataset.get_know_unknow_nodes()
            pickle.dump(args.best_total_y, f)
        args.fig_path = args.best_fig_path + args.exp_id + '/'
        if not os.path.exists(args.fig_path):
            os.makedirs(args.fig_path)
        read_out({'test': data_loader}, args.fig_path, creterions, args.exp_id, args)
