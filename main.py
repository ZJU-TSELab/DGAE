import torch
from torch import nn
import pytorch_lightning as pl
from data_provider.data_factory import data_provider
from args import initialize_args
from utils.logtool import print_config, print_header, print_train, set_wandb
from experiments import train_model, inference_model
from utils.evaluation import SlitPhyLoss, rmse_loss
from omegaconf import OmegaConf
from model._model_register import ModelRegistry
import numpy as np
import os
import FeaturePropagation

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
if __name__ == "__main__":
    args = initialize_args()
    args.exp_id = f'{args.input_len}-{args.pred_len}-{args.look_back}-{args.slide_step}-{args.input_features}-{args.target_features}-t_mark-{args.t_mark}-miss_mark-{args.miss_mark}-seed-{args.seed}'
    if args.unknown_nodes_path:
        per = args.unknown_nodes_path.split('/')[-1].split('_')[-1].split('.')[0]
        if '%' in per:
            args.exp_id = f'{per}-{args.exp_id}'
    args.exp_id = f'{args.dloader_name}-{args.exp_id}'
    if args.fp_step > 0:
        args.exp_id = f'fp_step-{args.fp_step}-{args.exp_id}'
    args.exp_id = f'{args.model}-{args.exp_id}'
    if args.transfer_only:
        args.exp_id = f'transferonly-{args.exp_id}'
    wandb = set_wandb(args)
    configs = OmegaConf.create(vars(args))
    pl.seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    slide_step = args.slide_step
    args.slide_step = 1  # use 1 for train to collect more data
    train_loader = data_provider(args, 'train')
    args.slide_step = slide_step
    val_loader = data_provider(args, 'val')
    test_loader = data_provider(args, 'test')
    data_loaders = dict(
        zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]))
    if args.model == 'DEFP4D':
        FeaturePropagation.inference(test_loader, args, 'DEFP4D')
        exit()
    elif args.model == 'DEFP':
        FeaturePropagation.inference(test_loader, args, 'DEFP')
        exit()
    loss_candidate = dict(
        zip([
            'mse',
            'huber',
            'mae',
            'rmse',
        ], [
            nn.MSELoss(),
            nn.HuberLoss(),
            nn.L1Loss(),
            rmse_loss,
        ]))
    loss = loss_candidate[args.loss]
    criterions = dict(
        zip([args.loss, 'split_phy_loss'],
            [loss, SlitPhyLoss(features=args.target_features)]))
    model = ModelRegistry.get_model(args.model)(args).to(args.device).float()
    if args.test_only:
        args.best_checkpoint_path = args.best_checkpoint_path + args.exp_id + '/'
        try:
            model_dict = torch.load(args.best_checkpoint_path +
                                    'best_val_checkpoint.pth')['state_dict']
            print(f'load pretrained from {args.best_checkpoint_path}')
        except FileNotFoundError:
            model_dict = torch.load(args.best_checkpoint_path +
                                    'best_train_checkpoint.pth')['state_dict']
            print(f'load pretrained from {args.best_checkpoint_path}')
        model.load_state_dict(model_dict)
        model.to(args.device).float()
        args.slide_step = slide_step
        inference_model(model, test_loader, args, criterions)
        exit()
    if args.transfer_only:
        model_dict = torch.load(args.pretrained_model_path)['state_dict']
        model.load_state_dict(model_dict)
        model.to(args.device).float()
        args.slide_step = slide_step
        inference_model(model, test_loader, args, criterions)
        exit()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    args.num_params = sum([np.prod(p.size()) for p in model_parameters])
    optimizers = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }
    optimizer = optimizers[args.optimizer](model.parameters(),
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)
    schdulers = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'step': torch.optim.lr_scheduler.StepLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'multistep': torch.optim.lr_scheduler.MultiStepLR
    }
    if args.scheduler == 'plateau':
        scheduler = schdulers[args.scheduler](optimizer,
                                              mode='min',
                                              factor=0.5,
                                              min_lr=1e-5,
                                              patience=args.scheduler_patience)
    elif args.scheduler == 'cosine':
        scheduler = schdulers[args.scheduler](optimizer, T_max=5, eta_min=1e-7)
    elif args.scheduler == 'step':
        scheduler = schdulers[args.scheduler](
            optimizer,
            step_size=args.scheduler_patience,
            gamma=0.2,
        )
    elif args.scheduler == 'multistep':
        scheduler = schdulers[args.scheduler](
            optimizer,
            milestones=args.scheduler_patience,
            gamma=0.5,
        )
        print(f'-> milestones: {scheduler.milestones}')
    else:

        class NullScheduler:

            def step(self):
                pass

        scheduler = NullScheduler()
    if args.verbose:
        print_header('*** Model Configurations ***')
        print(model)
    print_header('*** Experiment Configurations *** ')
    print_config(configs)
    print_train(args)
    print(f'├── train data dim: {len(train_loader.dataset)}')  #type:ignore
    print(f'├── val   data dim: {len(val_loader.dataset)}')  #type:ignore
    print(f'└── test  data dim: {len(test_loader.dataset)}')  #type:ignore
    pretrained = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loaders=data_loaders,
        criterions=criterions,
        args=args,
        wandb=wandb,
    )
    if args.save_model:
        import os
        if not os.path.exists('./pretrained'):
            os.makedirs('./pretrained')
        torch.save(pretrained, f'./pretrained/{args.exp_id}.pth')
    print("-> Done!")
