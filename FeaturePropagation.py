import torch
from torch import Tensor
from utils.evaluation import SlitPhyLoss
import pickle
import os


class DEFP4D(torch.nn.Module):

    def __init__(self, num_iterations: int):
        super(DEFP4D, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1).permute(0, 2, 1)
        dm = self.diffusion_adj(adj)
        out = torch.zeros_like(x)
        out[x != 0] = x[x != 0]
        out[x == 0] = x.mean()
        gap = 100
        i = 0
        while gap > 1e-10 and i <= self.num_iterations:
            # Diffuse current features
            last_out = out.clone()
            out = torch.matmul(dm, out)
            # Reset original known features
            # out[:, known_nodes] = x[:, known_nodes]
            out[x != 0] = x[x != 0]
            gap = torch.abs(out - last_out).mean()
            i += 1
        if gap <= 1e-10:
            print(f"Converged in {i} iterations.")
        else:
            print(f"Did not converge in {self.num_iterations} iterations.")
        out = out.permute(0, 2, 1).unsqueeze(-1)
        return out

    def diffusion_adj(self, adj: Tensor) -> Tensor:
        """convert dense adjacency matrix to sparse diffusion kernel

        Args:
            adj (Tensor): adjacency matrix,shape (N, N)
        Returns:
            diffusion matrix (Tensor): diffusion matrix, shape (N, N)
        """
        # 计算度矩阵 D
        out_deg = torch.diag(adj.sum(dim=1))
        in_deg = torch.diag(adj.sum(dim=0))
        deg = (out_deg + in_deg)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        # 计算 DM
        diff_matrix = deg_inv_sqrt @ (adj + adj.T)
        return diff_matrix


class DEFP(DEFP4D):

    def __init__(self, num_iterations: int):
        super(DEFP, self).__init__(num_iterations)

    def diffusion_adj(self, adj: Tensor) -> torch.Tensor:
        adj_sym = (adj + adj.T) / 2
        degree_matrix = torch.diag(adj_sym.sum(dim=1))
        degree_matrix_inv_sqrt = degree_matrix.pow(-0.5)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0
        dm = degree_matrix_inv_sqrt @ adj_sym @ degree_matrix_inv_sqrt
        return dm


def inference(data_loader, args, model_name='DEFP4D'):
    if model_name == 'DEFP4D':
        model = DEFP4D(args.fp_step)
    else:
        model = DEFP(args.fp_step)
    model.to(args.device)
    ys = []
    y_hats = []
    for _, batch in enumerate(data_loader):
        batch = [x.float().to(args.device) for x in batch]
        x_enc, x_dec, _, _ = batch
        y = x_dec[:, -args.pred_len:].clone()
        x_dec = x_dec[:, -args.pred_len:]
        adj, _ = data_loader.dataset.get_adj_and_pos_mark()
        adj = torch.from_numpy(adj).float().to(args.device)
        x_enc[torch.isnan(x_enc)] = 0  #在源数据集中，有些异常值被标注为0，这些异常值不参加任何评价计算
        _, unknown_nodes = data_loader.dataset.get_know_unknow_nodes()
        y_hat = model(x_enc, adj)
        y = y.detach().cpu()
        y_hat = y_hat.detach().cpu()
        ys.append(y)
        y_hats.append(y_hat)
    y = torch.cat(ys, dim=0)[:, :, unknown_nodes]
    y_hat = torch.cat(y_hats, dim=0)[:, :, unknown_nodes]
    metrics = SlitPhyLoss('v')
    miss_mark = torch.isnan(y)
    loss = metrics(y_hat, y, miss_mark)
    print(loss)
    print("-> save predictions")
    args.best_prediction_path = args.best_prediction_path + args.exp_id + '/'
    print(f'save to {args.best_prediction_path}')
    y = torch.cat(ys, dim=0)
    y_hat = torch.cat(y_hats, dim=0)
    best_total_y = {
        'true_trans': y,
        'pred_trans': y_hat,
    }
    best_total_y['known_nodes'], best_total_y[
        'unknown_nodes'] = data_loader.dataset.get_know_unknow_nodes()
    # path = f'./processed_data/{args.dloader_name}/fp/'
    if not os.path.exists(args.best_prediction_path):
        os.makedirs(args.best_prediction_path)
    with open(args.best_prediction_path + 'best_total_y.pkl', 'wb') as f:
        pickle.dump(best_total_y, f)
