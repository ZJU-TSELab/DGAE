import torch
from utils.logtool import detail_metrics
import random


def shared_step(model, dataloader, optimizer, criterions, args, mode, train, epoch):
    # Save step-wise metrics
    inverse_transform = dataloader.dataset.inverse_transform
    metrics = {}
    for metric_name, criterion in criterions.items():
        if metric_name == 'split_phy_loss':
            for sub_metric_name in criterion.sub_metrics:
                metrics[sub_metric_name] = 0.
        else:
            metrics[metric_name] = 0.
    metrics2 = metrics.copy()
    metrics3 = metrics.copy()
    if mode == 'train' and train == True:
        model.train()
        model.zero_grad()
        grad_enabled = True
    else:
        model.eval()
        grad_enabled = False

    # Save predictions
    total_y_true = []
    total_y_pred = []
    total_y_true_trans = []
    total_y_pred_trans = []
    # total_latents = []
    part_nodes = dict(
        zip(('known_nodes', 'unknown_nodes'), dataloader.dataset.get_know_unknow_nodes()))

    with torch.set_grad_enabled(grad_enabled):
        model.to(args.device)
        max_bacth_num = min(300, len(dataloader)) if mode == 'train' else len(dataloader)
        for batch_ix, batch in enumerate(dataloader):
            if batch_ix >= max_bacth_num:
                break
            batch = [x.float().to(args.device) for x in batch]
            x_enc, x_dec, enc_t_mark, dec_t_mark = batch
            if not args.t_mark:
                enc_t_mark = None
            y = x_dec[:, -args.pred_len:].clone()
            x_dec = x_dec[:, -args.pred_len:]
            adj, pos_mark = dataloader.dataset.get_adj_and_pos_mark()
            if not args.pos_mark:
                pos_mark = None
            adj = torch.from_numpy(adj).float().to(args.device)
            if pos_mark is not None:
                pos_mark = torch.from_numpy(pos_mark).float().to(args.device)
            x_enc[torch.isnan(x_enc)] = 0  #在源数据集中，有些异常值被标注为0，这些异常值不参加任何评价计算
            x_dec[torch.isnan(x_dec)] = 0
            if mode != 'test':
                batch_known_nodes = list(range(0, x_enc.size(2)))
                missing_ix = torch.ones_like(x_enc).to(args.device)
                batch_mask = random.sample(range(0, len(batch_known_nodes)),
                                           int(len(batch_known_nodes) * 0.25))
                batch_unmask = list(
                    set(range(0, len(batch_known_nodes))) - set(batch_mask))
                missing_ix[..., batch_mask, :] = 0
                x_enc = x_enc * missing_ix
            if mode != 'train':
                x_dec = None
            if mode == 'test':
                _, batch_mask = dataloader.dataset.get_know_unknow_nodes()
            if args.miss_mark:
                miss_mark = (x_enc == 0).float()
            else:
                miss_mark = None

            y_hat = model(adj, x_enc, batch_mask, enc_t_mark, miss_mark, pos_mark, x_dec,
                          epoch)
            loss_struc = 0
            if isinstance(y_hat, tuple):
                y_hat, *latents = y_hat
                if len(latents) == 2:
                    adj_free_hat, adj_cog_hat = latents
                    BCE = torch.nn.BCEWithLogitsLoss(reduction='mean')
                    adj_bi = (adj > 0).float()
                    loss_struc = BCE(adj_free_hat, adj_bi.expand(
                        adj_free_hat.size())) + BCE(adj_cog_hat,
                                                    adj_bi.T.expand(adj_cog_hat.size()))

            y_omi = torch.isnan(y)
            if (~y_omi).sum() == 0:
                continue
            loss = criterions[args.loss](y_hat[~y_omi], y[~y_omi])
            loss += loss_struc
            if grad_enabled:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            y = y.detach().cpu()
            y_hat = y_hat.detach().cpu()
            y_omi = y_omi.detach().cpu()
            inverse_y = inverse_transform(y)
            inverse_y_hat = inverse_transform(y_hat)
            total_y_true.append(y)
            total_y_pred.append(y_hat)
            total_y_true_trans.append(inverse_y)
            total_y_pred_trans.append(inverse_y_hat)
            # total_latents.append(latent.detach().cpu())
            if mode != 'test':
                metrics = detail_metrics(dataloader, criterions, args, mode + '_total ',
                                         metrics, y_omi, loss, inverse_y, inverse_y_hat,
                                         max_bacth_num, grad_enabled, batch_ix)
                for i, mask in enumerate([batch_mask, batch_unmask]):
                    inverse_y_hat_part = inverse_y_hat[..., mask, :]
                    inverse_y_part = inverse_y[..., mask, :]
                    y_hat_part = y_hat[..., mask, :]
                    y_part = y[..., mask, :]
                    y_omi_part = torch.isnan(inverse_y_part)
                    loss = criterions[args.loss](y_hat_part[~y_omi_part],
                                                 y_part[~y_omi_part])
                    if i == 0:
                        metrics2 = detail_metrics(dataloader, criterions, args,
                                                  mode + '_mask  ', metrics2, y_omi_part,
                                                  loss, inverse_y_part,
                                                  inverse_y_hat_part, max_bacth_num,
                                                  grad_enabled, batch_ix)
                    else:
                        metrics3 = detail_metrics(dataloader, criterions, args,
                                                  mode + '_unmask', metrics3, y_omi_part,
                                                  loss, inverse_y_part,
                                                  inverse_y_hat_part, max_bacth_num,
                                                  grad_enabled, batch_ix)
    total_y = {
        'true': torch.cat(total_y_true, dim=0),
        'pred': torch.cat(total_y_pred, dim=0),
        'true_trans': torch.cat(total_y_true_trans, dim=0),
        'pred_trans': torch.cat(total_y_pred_trans, dim=0),
        # 'latent': torch.cat(total_latents, dim=0)
    }

    if mode != 'test':
        for metric_name in metrics.keys():
            metrics[metric_name] /= batch_ix + 1  # average over batches
        for metric_name in metrics2.keys():
            metrics2[metric_name] /= batch_ix + 1
        for metric_name in metrics3.keys():
            metrics3[metric_name] /= batch_ix + 1

    else:
        for name, nodes in part_nodes.items():
            if len(nodes) == 0:
                continue
            inverse_y_hat_part = total_y['pred_trans'][..., nodes, :]
            inverse_y_part = total_y['true_trans'][..., nodes, :]
            y_hat_part = total_y['pred'][..., nodes, :]
            y_part = total_y['true'][..., nodes, :]
            y_omi_part = torch.isnan(inverse_y_part)
            loss = criterions[args.loss](y_hat_part[~y_omi_part], y_part[~y_omi_part])
            if name == 'known_nodes':
                metrics = detail_metrics(dataloader, criterions, args, mode + f'_{name}',
                                         metrics, y_omi_part, loss, inverse_y_part,
                                         inverse_y_hat_part, max_bacth_num, grad_enabled,
                                         batch_ix)
            elif name == 'unknown_nodes':
                metrics2 = detail_metrics(dataloader, criterions, args, mode + f'_{name}',
                                          metrics2, y_omi_part, loss, inverse_y_part,
                                          inverse_y_hat_part, max_bacth_num, grad_enabled,
                                          batch_ix)
    return model, (metrics, metrics2, metrics3), total_y
