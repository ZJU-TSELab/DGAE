import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse
from torch import Tensor
from torch import nn


def diffusion_adj(adj: Tensor) -> Tensor:
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


def diffusion_adj2(adj: Tensor) -> Tensor:
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


def diffusion_adj3(adj: Tensor) -> Tensor:
    """convert dense adjacency matrix to sparse diffusion kernel

    Args:
        adj (Tensor): adjacency matrix,shape (N, N)
    Returns:
        diffusion matrix (Tensor): diffusion matrix, shape (N, N)
    """
    # 计算度矩阵 D
    deg = torch.diag(adj.sum(dim=1))
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    # 计算 DM
    diff_matrix = deg_inv_sqrt @ adj
    return diff_matrix


class SpDiff(MessagePassing):

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 alpha=1.,
                 t=1,
                 remove_self_loop=True) -> None:
        """_summary_

        Args:
            input_dim (int): input feature dimension
            out_dim (int): output feature dimension
            beta (float): diffusion rate
            t (int): diffusion steps
            remove_self_loop (bool, optional): remove self-loop or not. Defaults to True.
        """
        super(SpDiff, self).__init__(aggr='add')  # 使用加法聚合
        self.lin = nn.Linear(input_dim, out_dim)
        self.act = nn.ReLU()
        self.remove_self_loop = remove_self_loop
        # 设置每步的可学习扩散率
        # self.betas = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(beta)) for _ in range(t + 1)])
        self.alpha = alpha
        self.t = t
        self.edge = None

    def forward(self, x, adj: Tensor) -> Tensor:
        # x: 节点特征 (batch_size, num_nodes, in_channels)
        # edge_index: 边索引 (2, num_edges)
        # edge_weight: 边权重 (num_edges, )
        self.edge_index, self.edge_weight = self.adj_to_sparse_diff_kernel(adj)
        return self.propagate(self.edge_index,
                              x=x,
                              edge_weight=self.edge_weight)

    def message(self, x_j, edge_weight):
        # 直接返回邻居消息，乘以边权重
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 聚合函数返回邻居消息的加权平均
        return self.act(self.lin(aggr_out))  #  # 直接返回聚合结果

    def adj_to_sparse_diff_kernel(self, adj: Tensor) -> tuple[Tensor, Tensor]:
        """convert dense adjacency matrix to sparse diffusion kernel

        Args:
            adj (Tensor): adjacency matrix,shape (N, N)
        Returns:
            tuple[Tensor, Tensor]: edge_index, edge_weight
        """

        # 计算度矩阵 D
        if self.remove_self_loop:
            adj[torch.eye(adj.size(0), dtype=torch.bool)] = 0
        P = diffusion_adj(adj)
        # 计算扩散核 K
        if not self.remove_self_loop:
            K = torch.eye(adj.size(0), device=adj.device)
        else:
            K = torch.zeros_like(adj, device=adj.device)
        # for i in range(1, self.t + 1):
        #     K += (self.alpha**i) * torch.linalg.matrix_power(P, i)
        for i in range(1, self.t + 1):  #从0开始意味着加了一个原始数据，这或许不利于重构任务。
            K += self.alpha * (1 - self.alpha)**(
                i - 1) * torch.linalg.matrix_power(P, i)
        edge_index, edge_weight = dense_to_sparse(
            K.T)  #注意要转置,因为MPNN默认是A.T@X而不是A@X,所以这里先把A转置过来，以保证MPNN输出A@X

        return edge_index, edge_weight

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  (MPNN): DiffusionKernel(alpha={self.alpha}, k={self.t})\n'
                f'  (lin): {self.lin}\n'
                f'  (act): {self.act}\n'
                f'  (remove_self_loop): {self.remove_self_loop}\n'
                f')')


class SpDiff3(MessagePassing):

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 alpha=1.,
                 t=1,
                 remove_self_loop=True) -> None:
        """_summary_

        Args:
            input_dim (int): input feature dimension
            out_dim (int): output feature dimension
            beta (float): diffusion rate
            t (int): diffusion steps
            remove_self_loop (bool, optional): remove self-loop or not. Defaults to True.
        """
        super(SpDiff3, self).__init__(aggr='add')  # 使用加法聚合
        self.lin = nn.Linear(input_dim, out_dim)
        self.act = nn.ReLU()
        self.remove_self_loop = remove_self_loop
        # 设置每步的可学习扩散率
        # self.betas = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(beta)) for _ in range(t + 1)])
        self.alpha = alpha
        self.t = t
        self.edge = None

    def forward(self, x, adj: Tensor) -> Tensor:
        # x: 节点特征 (batch_size, num_nodes, in_channels)
        # edge_index: 边索引 (2, num_edges)
        # edge_weight: 边权重 (num_edges, )
        self.edge_index, self.edge_weight = self.adj_to_sparse_diff_kernel(adj)
        return self.propagate(self.edge_index,
                              x=x,
                              edge_weight=self.edge_weight)

    def message(self, x_j, edge_weight):
        # 直接返回邻居消息，乘以边权重
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 聚合函数返回邻居消息的加权平均
        return self.act(self.lin(aggr_out))  #  # 直接返回聚合结果

    def adj_to_sparse_diff_kernel(self, adj: Tensor) -> tuple[Tensor, Tensor]:
        """convert dense adjacency matrix to sparse diffusion kernel

        Args:
            adj (Tensor): adjacency matrix,shape (N, N)
        Returns:
            tuple[Tensor, Tensor]: edge_index, edge_weight
        """

        # 计算度矩阵 D
        if self.remove_self_loop:
            adj[torch.eye(adj.size(0), dtype=torch.bool)] = 0
        P = diffusion_adj3(adj)
        # 计算扩散核 K
        if not self.remove_self_loop:
            K = torch.eye(adj.size(0), device=adj.device)
        else:
            K = torch.zeros_like(adj, device=adj.device)
        # for i in range(1, self.t + 1):
        #     K += (self.alpha**i) * torch.linalg.matrix_power(P, i)
        for i in range(1, self.t + 1):  #从0开始意味着加了一个原始数据，这或许不利于重构任务。
            K += self.alpha * (1 - self.alpha)**(
                i - 1) * torch.linalg.matrix_power(P, i)
        edge_index, edge_weight = dense_to_sparse(
            K.T)  #注意要转置,因为MPNN默认是A.T@X而不是A@X,所以这里先把A转置过来，以保证MPNN输出A@X

        return edge_index, edge_weight

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  (MPNN): DiffusionKernel(alpha={self.alpha}, k={self.t})\n'
                f'  (lin): {self.lin}\n'
                f'  (act): {self.act}\n'
                f'  (remove_self_loop): {self.remove_self_loop}\n'
                f')')


class BiSpDiff(torch.nn.Module):

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 alpha=1.,
                 t=1,
                 remove_self_loop=True) -> None:
        super(BiSpDiff, self).__init__()
        self.gnn = SpDiff(input_dim, out_dim, alpha, t, remove_self_loop)
        self.gnn_r = SpDiff(input_dim, out_dim, alpha, t, remove_self_loop)
        self.input_dim = input_dim
        # self.fc = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU())

    def forward(self, x, adj1: Tensor, adj2=None):
        if adj2 is None:
            adj2 = adj1.T
        if x.size(-1) != self.input_dim:
            x1, x2 = x.chunk(2, dim=-1)
            x1 = self.gnn(x1, adj1)
            x2 = self.gnn_r(x2, adj2)
        else:
            x1 = self.gnn(x, adj1)
            x2 = self.gnn_r(x, adj2)
        return x1 + x2


class BiSpDiff3(torch.nn.Module):

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 alpha=1.,
                 t=1,
                 remove_self_loop=True) -> None:
        super(BiSpDiff3, self).__init__()
        self.gnn = SpDiff3(input_dim, out_dim, alpha, t, remove_self_loop)
        self.gnn_r = SpDiff3(input_dim, out_dim, alpha, t, remove_self_loop)
        self.input_dim = input_dim
        # self.fc = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU())

    def forward(self, x, adj1: Tensor, adj2=None):
        if adj2 is None:
            adj2 = adj1.T
        if x.size(-1) != self.input_dim:
            x1, x2 = x.chunk(2, dim=-1)
            x1 = self.gnn(x1, adj1)
            x2 = self.gnn_r(x2, adj2)
        else:
            x1 = self.gnn(x, adj1)
            x2 = self.gnn_r(x, adj2)
        return x1 + x2


class BiSpDiff2(torch.nn.Module):

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 alpha=1.,
                 t=1,
                 remove_self_loop=True) -> None:
        super(BiSpDiff2, self).__init__()
        self.gnn = SpDiff(input_dim, out_dim, alpha, t, remove_self_loop)

    def forward(self, x, adj1: Tensor, adj2=None):
        if adj2 is None:
            adj2 = adj1.T
        x1 = self.gnn(x, adj1)
        x2 = self.gnn(x, adj2)
        return x1 + x2


class JointDiff(MessagePassing):

    def __init__(self,
                 input_dim: int,
                 out_dim: int,
                 alpha=1.,
                 t=1,
                 remove_self_loop=True) -> None:
        """_summary_

        Args:
            input_dim (int): input feature dimension
            out_dim (int): output feature dimension
            beta (float): diffusion rate
            t (int): diffusion steps
            remove_self_loop (bool, optional): remove self-loop or not. Defaults to True.
        """
        super(JointDiff, self).__init__(aggr='add')  # 使用加法聚合
        self.lin = nn.Linear(input_dim, out_dim)
        self.act = nn.ReLU()
        self.remove_self_loop = remove_self_loop
        # 设置每步的可学习扩散率
        # self.betas = nn.ParameterList(
        #     [nn.Parameter(torch.tensor(beta)) for _ in range(t + 1)])
        self.alpha = alpha
        self.t = t
        self.edge = None

    def forward(self, x, adj: Tensor) -> Tensor:
        # x: 节点特征 (batch_size, num_nodes, in_channels)
        # edge_index: 边索引 (2, num_edges)
        # edge_weight: 边权重 (num_edges, )
        self.edge_index, self.edge_weight = self.adj_to_sparse_diff_kernel(adj)
        return self.propagate(self.edge_index,
                              x=x,
                              edge_weight=self.edge_weight)

    def message(self, x_j, edge_weight):
        # 直接返回邻居消息，乘以边权重
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 聚合函数返回邻居消息的加权平均
        return self.act(self.lin(aggr_out))  #  # 直接返回聚合结果

    def adj_to_sparse_diff_kernel(self, adj: Tensor) -> tuple[Tensor, Tensor]:
        """convert dense adjacency matrix to sparse diffusion kernel

        Args:
            adj (Tensor): adjacency matrix,shape (N, N)
        Returns:
            tuple[Tensor, Tensor]: edge_index, edge_weight
        """

        # 计算度矩阵 D
        if self.remove_self_loop:
            adj[torch.eye(adj.size(0), dtype=torch.bool)] = 0
        P = diffusion_adj2(adj)
        # 计算扩散核 K
        if not self.remove_self_loop:
            K = torch.eye(adj.size(0), device=adj.device)
        else:
            K = torch.zeros_like(adj, device=adj.device)
        # for i in range(1, self.t + 1):
        #     K += (self.alpha**i) * torch.linalg.matrix_power(P, i)
        for i in range(1, self.t + 1):  #从0开始意味着加了一个原始数据，这或许不利于重构任务。
            K += self.alpha * (1 - self.alpha)**(
                i - 1) * torch.linalg.matrix_power(P, i)
        edge_index, edge_weight = dense_to_sparse(
            K.T)  #注意要转置,因为MPNN默认是A.T@X而不是A@X,所以这里先把A转置过来，以保证MPNN输出A@X

        return edge_index, edge_weight

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  (MPNN): DiffusionKernel(alpha={self.alpha}, k={self.t})\n'
                f'  (lin): {self.lin}\n'
                f'  (act): {self.act}\n'
                f'  (remove_self_loop): {self.remove_self_loop}\n'
                f')')
