import copy
import os.path as osp

import torch
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor
from graph_sample.fastgae_sampling import (get_distribution, node_sampling)
from util.samp_graph import gen_subgraph


class AttributedGSSampler(object):
    r"""
    Args:
        data (torch_geometric.data.Data): The graph data object.
        batch_size (int): The approximate number of samples per batch to load.
        num_steps (int, optional): The number of iterations.
            (default: :obj:`1`)
        sample_coverage (int): How many samples per node should be used to
            compute normalization statistics. (default: :obj:`50`)
        save_dir (string, optional): If set, will save normalization
            statistics to the :obj:`save_dir` directory for faster re-use.
            (default: :obj:`None`)

        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
    """
    def __init__(self, data, batch_size, num_steps=1, sample_coverage=50,
                 save_dir=None, log=True, reload=False, cal_norm=True):
        """
        :param data:
        :param batch_size:
        :param num_steps:
        :param sample_coverage:
        :param save_dir:
        :param log:
        """

        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                value=data.edge_attr, sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None
        self.data.edge_attr = None

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.sample_coverage = sample_coverage
        self.log = log
        self.__count__ = 0
        self.cal_norm = cal_norm

        if cal_norm:
            path = osp.join(save_dir or '', self.__filename__)
            if save_dir is not None and osp.exists(path) and reload==False:  # pragma: no cover
                self.node_norm, self.edge_norm, self.edge_sample_norm = torch.load(path)
            else:
                self.node_norm, self.edge_norm, self.edge_sample_norm = self.__compute_norm__()
                if save_dir is not None:  # pragma: no cover
                    torch.save((self.node_norm, self.edge_norm, self.edge_sample_norm), path)
        else:
            self.node_norm, self.edge_norm, self.edge_sample_norm = None, None, None

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self.sample_coverage}.pt'


    def __sample_nodes__(self, num_examples):
        raise NotImplementedError


    def __sample__(self, num_examples):
        node_samples = self.__sample_nodes__(num_examples)

        samples = []
        for node_idx in node_samples:

            node_idx = node_idx.unique()
            adj, edge_idx = self.adj.saint_subgraph(node_idx)
            samples.append((node_idx, edge_idx, adj))
        return samples

    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description('Compute AttributedGSSampler normalization')
        # num_samples含义， total_sampled_nodes含义（所有被sample的节点的个数和）
        num_samples = total_sampled_nodes = 0

        while total_sampled_nodes < self.N * self.sample_coverage:
            num_sampled_nodes = 0
            samples = self.__sample__(50)
            for node_idx, edge_idx, _ in samples:
                node_count[node_idx] += 1
                edge_count[edge_idx] += 1
                num_sampled_nodes += node_idx.size(0)
            total_sampled_nodes += num_sampled_nodes
            num_samples += 50

            if self.log:  # pragma: no cover
                pbar.update(num_sampled_nodes)

        if self.log:  # pragma: no cover
            pbar.close()

        row, col, _ = self.adj.coo()

        edge_norm = (node_count[col] / edge_count).clamp_(0, 1e4)
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        edge_count[edge_count == 0] = 0.1
        edge_sample_norm = num_samples / edge_count / self.E
        return node_norm, edge_norm, edge_sample_norm


    def __get_data_from_sample__(self, sample):
        node_idx, edge_idx, adj = sample

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, value = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        for key, item in self.data:
            if item.size(0) == self.N:
                data[key] = item[node_idx]
            elif item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        if self.cal_norm:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]
            data.edge_sample_norm = self.edge_sample_norm[edge_idx]

        return data

    def __next__(self):
        if self.__count__ < len(self):
            self.__count__ += 1
            sample = self.__sample__(1)[0]
            data = self.__get_data_from_sample__(sample)
            return data
        else:
            raise StopIteration

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        self.__count__ = 0
        return self




class AGSRandomWalkSampler(AttributedGSSampler):
    r"""
    Args:
        batch_size (int): The number of walks to sample per batch.
        walk_length (int): The length of each random walk.
    """
    def __init__(self, data, adj_dic, adj_sim_dic, measure, alpha, batch_size, walk_length, num_steps=1,
                 sample_coverage=50, save_dir=None, log=True, reload=False, cal_norm=True):
        """
        :param data:            pyG data
        :param adj_dic:        dic
        :param adj_dic_sim:    dic
        :param measure:         str uniform, degree, core: sample root list的方法
        :param alpha:           int, degree/core
        :param batch_size:      root size
        :param walk_length:     rw length
        :param num_steps:
        :param sample_coverage:
        :param save_dir:
        :param log:
        :param reload:
        :param cal_norm:
        """
        self.adj_dic, self.adj_sim_dic = adj_dic, adj_sim_dic
        self.walk_length = walk_length

        self.measure = measure
        self.alpha = alpha
        assert data.edge_index is not None
        self.N = N = data.num_nodes
        self.sp_adj = sp.coo_matrix((data.edge_attr, (data.edge_index[0], data.edge_index[1])), shape=(N, N))
        self.node_distribution = get_distribution(self.measure, self.alpha, self.sp_adj)
        super(AGSRandomWalkSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, reload, cal_norm=cal_norm)

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __sample_nodes__(self, num_examples):
        """
        :param num_examples: number of subgraphs
        :return: list, node graphs
        """
        node_sample =  []
        for _ in range(num_examples):
            if self.measure == 'uniform':
                roots = np.random.choice(a=self.N, size=self.batch_size, replace=False)
            else:
                roots = node_sampling(adj=self.sp_adj, distribution=self.node_distribution,
                                      nb_node_samples=self.batch_size, replace=False)
            subg_nodes = gen_subgraph(roots, self.walk_length, self.adj_dic, self.adj_sim_dic)

            node_sample.append(torch.from_numpy(subg_nodes))
        return node_sample


