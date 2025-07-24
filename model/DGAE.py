import torch
from torch import nn
from torch import Tensor
from einops import rearrange
from layers.DirichletDiff import BiSpDiff
from model._model_register import ModelRegistry


class FeaturePropagation(torch.nn.Module):

    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                known_nodes: list) -> torch.Tensor:
        dm = self.diffusion_adj(adj)
        out = torch.zeros_like(x)
        out[:, known_nodes] = x[:, known_nodes]
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.matmul(dm, out)
            # Reset original known features
            out[:, known_nodes] = x[:, known_nodes]
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
        diff_matrix = deg_inv_sqrt @ adj
        return diff_matrix


class DGAE(nn.Module):

    def __init__(self,
                 input_dim,
                 latent_dim,
                 out_dim=None,
                 diff_alpha=0.8,
                 k_hop=2,
                 fp_step=8):
        super(DGAE, self).__init__()
        if out_dim is None:
            out_dim = input_dim

        delta_dim = (latent_dim - input_dim) // 3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim + delta_dim) * 2), nn.ReLU(),
            BiSpDiff(input_dim + delta_dim,
                     input_dim + 2 * delta_dim,
                     alpha=diff_alpha,
                     t=k_hop,
                     remove_self_loop=False),
            nn.Linear(input_dim + 2 * delta_dim, latent_dim))

        delta_dim = (out_dim - latent_dim) // 3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (latent_dim + delta_dim) * 2), nn.ReLU(),
            BiSpDiff(latent_dim + delta_dim,
                     latent_dim + 2 * delta_dim,
                     alpha=diff_alpha,
                     t=k_hop,
                     remove_self_loop=False),
            nn.Linear(latent_dim + 2 * delta_dim, out_dim))
        self.intepolation = FeaturePropagation(num_iterations=fp_step)

    def encode(self, x, adj, encoder):
        for layer in encoder:
            if isinstance(layer, BiSpDiff):
                x = layer(x, adj)
            else:
                x = layer(x)
        return x

    def decode(self, x, adj, decoder):
        for layer in decoder:
            if isinstance(layer, BiSpDiff):
                x = layer(x, adj)
            else:
                x = layer(x)
        return x

    def forward(self, x, adj, known_nodes: list):

        x_known = x[..., known_nodes, :]
        adj_known = adj[known_nodes, :][:, known_nodes]
        latent_known = self.encode(x_known, adj_known, self.encoder)
        latent_full_cog = torch.zeros(x.size(0),
                                      x.size(1),
                                      latent_known.size(-1),
                                      device=x.device)
        latent_full_free = torch.zeros(x.size(0),
                                       x.size(1),
                                       latent_known.size(-1),
                                       device=x.device)
        latent_full_cog[..., known_nodes, :] = latent_known
        latent_full_free[..., known_nodes, :] = latent_known
        latent_full_cog = self.intepolation(latent_full_cog, adj.T, known_nodes)
        latent_full_free = self.intepolation(latent_full_free, adj, known_nodes)
        latent_full = latent_full_cog + latent_full_free
        x_recon = self.decode(latent_full, adj, self.decoder)
        return x_recon, latent_full


@ModelRegistry.register(name='DGAE')
class Model(nn.Module):

    def __init__(self, configs) -> None:
        super(Model, self).__init__()

        self.mgae = DGAE(input_dim=configs.input_len *
                         (1 + int(configs.t_mark) + int(configs.miss_mark)),
                         latent_dim=configs.latent_dim,
                         out_dim=configs.input_len,
                         diff_alpha=configs.diff_alpha,
                         k_hop=configs.k_hop,
                         fp_step=configs.fp_step)
        self.miss_embedding = nn.Linear(configs.input_len, configs.input_len)
        self.time_embedding_1 = nn.Embedding(60, configs.input_len // 3)  #mins of hour
        self.time_embedding_2 = nn.Embedding(24, configs.input_len // 3)  #hour of day
        self.time_embedding_3 = nn.Embedding(7, configs.input_len -
                                             configs.input_len // 3 * 2)  #day of week

    def forward(self, adj, x, unknown_nodes: list, t_mark=None, miss_mark=None, *args):
        adj = adj.T
        x = rearrange(x, 'b l n c -> b n (l c)')
        known_nodes = list(set(range(x.size(1))) - set(unknown_nodes))
        if miss_mark is not None:
            miss_mark = rearrange(miss_mark, 'b l n c -> b n (l c)')
            miss_mark = self.miss_embedding(miss_mark)
            x = torch.cat([x, miss_mark], dim=-1)
        if t_mark is not None:
            t_mark = t_mark.long()  #type: ignore
            time_embedding = torch.cat((self.time_embedding_1(
                t_mark[:, -1:, 0]), self.time_embedding_2(
                    t_mark[:, -1:, 1]), self.time_embedding_3(t_mark[:, -1:, 2])),
                                       dim=-1).expand(x.size(0), x.size(1), -1)
            x = torch.cat([x, time_embedding], dim=-1)
        x_recon, latent = self.mgae(x, adj, known_nodes)
        x_recon = rearrange(x_recon, 'b n (l c) -> b l n c', c=1)
        return x_recon, latent
