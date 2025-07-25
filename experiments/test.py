import torch
from utils.logtool import detail_metrics
from sklearn.metrics import r2_score


def read_out(data_loaders, fig_path, criterions, exp_id, args, epoch=-1):
    part_nodes = {
        'known_nodes': args.best_total_y['known_nodes'],
        'unknown_nodes': args.best_total_y['unknown_nodes'],
    }

    for i, mode in enumerate(data_loaders.keys()):
        metrics = {}
        for metric_name, criterion in criterions.items():
            if metric_name == 'split_phy_loss':
                for sub_metric_name in criterion.sub_metrics:
                    metrics[sub_metric_name] = 0
            else:
                metrics[metric_name] = 0
        metrics['R2'] = 0
        metrics2 = metrics.copy()
        if 'val' in args.stop_based and mode == 'train':
            continue
        elif 'train' in args.stop_based and mode == 'val':
            continue
        y = args.best_total_y[mode]['true']
        y_hat = args.best_total_y[mode]['pred']
        inverse_y = args.best_total_y[mode]['true_trans']
        inverse_y_hat = args.best_total_y[mode]['pred_trans']

        y_omi = torch.isnan(y)
        for name, nodes in part_nodes.items():
            if len(nodes) == 0:
                continue
            inverse_y_hat_part = inverse_y_hat[..., nodes, :]
            inverse_y_part = inverse_y[..., nodes, :]
            y_hat_part = y_hat[..., nodes, :]
            y_part = y[..., nodes, :]
            y_omi_part = y_omi[..., nodes, :]
            if y.shape[-1] != y_hat.shape[-1]:
                loss = criterions[args.loss](y_hat_part, y_part, y_omi_part)
                y_hat_part = y_hat_part[..., -1:]
                inverse_y_hat_part = inverse_y_hat_part[..., -1:]
            else:
                loss = criterions[args.loss](y_hat_part[~y_omi_part], y_part[~y_omi_part])
            R2 = r2_score(y_true=inverse_y_part[~y_omi_part],
                          y_pred=inverse_y_hat_part[~y_omi_part])
            if name == 'known_nodes':
                metrics = detail_metrics(data_loaders[mode], criterions, args, f'{name}',
                                         metrics, y_omi_part, loss, inverse_y_part,
                                         inverse_y_hat_part)
                metrics['R2'] = R2
            elif name == 'unknown_nodes':
                metrics2 = detail_metrics(data_loaders[mode], criterions, args, f'{name}',
                                          metrics2, y_omi_part, loss, inverse_y_part,
                                          inverse_y_hat_part)
                metrics2['R2'] = R2

        with open(args.best_metrics_path + 'results.txt', 'a', encoding='utf-8') as f:
            description = f'├── exp_id: {exp_id} | Best Based on {mode:<5} Known Nodes Epoch: {epoch:<3}'
            for metric_name, metric in metrics.items():
                description += f' | {metric_name}: {metric:.4f}'
            f.write(description + '\n')
            print(description)
            description = f'├── exp_id: {exp_id} | Best Based on {mode:<5} Unknown Nodes Epoch : {epoch:<3}'
            for metric_name, metric in metrics2.items():
                description += f' | {metric_name}: {metric:.4f}'
            f.write(description + '\n')
            print(description)

            if i < len(data_loaders.keys()) - 1:
                f.write(f'{"-"*len(description)}' + '\n')
                print(f'{"-"*len(description)}')
    with open(args.best_metrics_path + 'results.txt', 'a', encoding='utf-8') as f:
        f.write(f'{"="*len(description)}' + '\n')
        print(f'{"="*len(description)}')
    print('-> Testing finished!')
