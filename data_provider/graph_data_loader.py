import numpy as np
from torch.utils.data import Dataset
import warnings
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class InductivePEMS(Dataset):

    def __init__(self,
                 data_path: str,
                 size: list[int],
                 flag: str,
                 scale: bool,
                 freq: str,
                 target_features: str,
                 input_features: str,
                 dloader_name: str,
                 splits: list,
                 step: int = 1,
                 inverse=False,
                 unknown_nodes_path='',
                 unknown_nodes_num=0,
                 sp_adj=False):
        assert flag in ['train', 'test', 'val'], 'flag should be train, test or val'
        fea_dict = {
            'q': [0],
            'o': [1],
            'v': [2],
            'qv': [0, 2],
            'qov': [0, 1, 2],
        }
        self.tar_feas = fea_dict[target_features]
        self.in_feas = fea_dict[input_features]
        self.flag = flag
        self.input_len = size[0]
        self.look_back = size[1]
        assert self.look_back <= self.input_len, 'look_back should be smaller than input_len'
        self.pred_len = size[2]
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.data_path = data_path
        self.unknown_nodes_num = unknown_nodes_num
        self.unknown_nodes_path = unknown_nodes_path
        self.dloader_name = dloader_name
        self.step = step
        self.train_ratio, self.valid_ratio = splits
        self.sp_adj = sp_adj
        self.__read_data__()

    def __read_data__(self):
        raw_data = np.load(self.data_path, allow_pickle=True)

        # features: timestamp shape (time_horizon,)
        self.time_mark = raw_data['time_stamp']
        if self.time_mark is not None:
            self.time_mark = time_features(self.time_mark, freq=self.freq)

        # features: lon,lat shape (num_nodes,features_dim)
        self.pos_mark = raw_data['pos_stamp']
        # shape (num_nodes,num_nodes)
        self.adj = raw_data['adj']  #based on distance
        if self.sp_adj:
            for i in range(len(self.adj)):
                for j in range(len(self.adj)):
                    if i == j:
                        continue
                    if self.adj[i, j] > self.adj[j, i]:
                        self.adj[j, i] = 0
                    else:
                        self.adj[i, j] = 0
        # time
        raw_data = raw_data['data']
        raw_data[raw_data <= 0] = np.nan
        if self.unknown_nodes_path:
            self.unknown_nodes = np.load(self.unknown_nodes_path)
            self.known_nodes = np.setdiff1d(range(raw_data.shape[1]), self.unknown_nodes)
        else:
            np.random.seed(0)
            self.unknown_nodes = np.random.choice(range(raw_data.shape[1]),
                                                  self.unknown_nodes_num,
                                                  replace=False)
            self.known_nodes = np.setdiff1d(range(raw_data.shape[1]), self.unknown_nodes)
        self.unknown_nodes = self.unknown_nodes.tolist()
        self.known_nodes = self.known_nodes.tolist()
        train_data = raw_data[:int(0.7 * len(raw_data))][:, self.known_nodes]
        if self.flag == 'train':
            self.data = train_data
            if self.time_mark is not None:
                self.time_mark = self.time_mark[:int(0.7 * len(self.time_mark))]
            if self.pos_mark is not None:
                self.pos_mark = self.pos_mark[self.known_nodes]
            self.adj = self.adj[self.known_nodes][:, self.known_nodes]
        elif self.flag == 'val':
            self.data = raw_data[:int(0.7 * len(raw_data))][:, self.known_nodes]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[:int(0.7 * len(self.time_mark))]
            if self.pos_mark is not None:
                self.pos_mark = self.pos_mark[self.known_nodes]
            self.adj = self.adj[self.known_nodes][:, self.known_nodes]
        else:
            self.data = raw_data[int(0.7 * len(raw_data)):]
            if self.time_mark is not None:
                self.time_mark = self.time_mark[int(0.7 * len(self.time_mark)):]
            # if self.pred_len == 0:
            self.data = self.data[self.input_len:]  #保持纯kriging任务跟prediction任务是在同一个测试集上评价
            if self.time_mark is not None:
                self.time_mark = self.time_mark[self.input_len:]
        if self.scale:
            self.mean_ = np.nanmean(train_data, axis=(0, 1), keepdims=True)
            self.std_ = np.nanstd(train_data, axis=(0, 1), keepdims=True)
            self.data = (self.data - self.mean_) / (self.std_)
        self.data_y = self.data.copy()
        if self.flag == 'test':
            self.data[:, self.unknown_nodes] = np.nan
        self.data = self.data[..., self.in_feas]
        self.data_y = self.data_y[..., self.tar_feas]

    def get_adj_and_pos_mark(self):
        return self.adj, self.pos_mark

    def get_know_unknow_nodes(self):
        return self.known_nodes, self.unknown_nodes

    def __getitem__(self, index):
        index = index * self.step
        input_begin = index
        input_end = input_begin + self.input_len
        output_begin = input_end - self.look_back
        output_end = input_end + self.pred_len
        seq_x = self.data[input_begin:input_end]
        seq_y = self.data_y[output_begin:output_end]
        x_t_mark, y_t_mark = np.zeros_like(seq_x), np.zeros_like(seq_y)
        if self.time_mark is not None:
            x_t_mark = self.time_mark[input_begin:input_end]
            y_t_mark = self.time_mark[output_begin:output_end]
        return seq_x, seq_y, x_t_mark, y_t_mark

    def __len__(self):
        return (len(self.data) - self.input_len -
                self.pred_len) // self.step + 1  #type:ignore

    def inverse_transform(self, y):
        return y * self.std_[..., self.tar_feas] + self.mean_[..., self.tar_feas]
