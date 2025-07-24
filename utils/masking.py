import torch


class TriangularCausalMask():

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool),
                                    diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():

    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1],
                           dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def find_taget_sensor(A, a):
    A = A + A.T
    a = torch.tensor(a, device=A.device)
    # 找到节点集 a 的一跳邻居节点集 b
    b = (A[a] > 0).nonzero(as_tuple=True)[1].unique()
    # 去除节点集 a 中的节点
    b = b[~b.isin(a)]
    # 找到节点集 b 的一跳邻居节点集 c
    c = (A[b] > 0).nonzero(as_tuple=True)[1].unique()
    # 去除节点集 a 和 b 中的节点
    c = c[~c.isin(torch.cat((a, b)))]
    all_nodes = torch.cat((a, b, c)).cpu().numpy().tolist()
    b = b.cpu().numpy().tolist()
    return all_nodes, b
