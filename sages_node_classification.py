# import os.path as osp
import torch
import random
import argparse
import numpy as np

import torch.nn.functional as F
from datasets.planetoid import Planetoid
from gnn.sages_nn import SAGES

from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GAE
from sklearn.cluster import SpectralClustering

from util.classifier import Classifier
# from graph_sample.saint_sampler import, FastGAENodeSampler
from graph_sample.gs_sampler import AGSRandomWalkSampler
from gsgae_preprocess import gen_graph_feat, gen_edge_sim, gen_adj_list
from util.cluster_metrics import clustering


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SAGES')
parser.add_argument('--dataset', type=str, default="Cora",
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--sample', type=str, default='GS')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--early_stop', type=int, default=500,
                    help='Number of Early Stop.')
parser.add_argument('--channels', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                            'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                   help='Normalization method for the adjacency matrix.')
args = parser.parse_args()

assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
kwargs = {'GAE': GAE, 'SAGES':SAGES}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(args.seed)
# 读入数据集
root_file = '/home/wangjialin/pyG_example/data/' + args.dataset
dataset = Planetoid(root=root_file, name=args.dataset)
data = dataset[0]

idx_train, idx_val, idx_test = [], [], []
for i in range(data.x.size(0)):
    if data.train_mask[i]:
        idx_train.append(i)
    elif data.val_mask[i]:
        idx_val.append(i)
    elif data.test_mask[i]:
        idx_test.append(i)

# 由于直接训练得到的node embedding存在nan和inf，需要处理
# 这里处理是把每一行中的nan和inf替换成为此行的embeddding的均值，输入的是numpy数组
def fill_ndarray(t1):
    for i in range(t1.shape[1]):  # 遍历每一列（每一列中的nan替换成该列的均值）
        temp_col = t1[:, i]  # 当前的一列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:  # 不为0，说明当前这一列中有nan
            temp_not_nan_col = temp_col[temp_col == temp_col]  # 去掉nan的ndarray

            # 选中当前为nan的位置，把值赋值为不为nan的均值
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()  # mean()表示求均值。
    return t1



# 设置谱聚类，进行测试
if args.dataset == 'Cora':
    n_clusters = 7
    Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
elif args.dataset == 'CiteSeer':
    n_clusters = 6
    Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
elif args.dataset == 'PubMed':
    n_clusters = 3
    Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)

# 开始处理有关子图采样的部分

# 计算线性GCN核后的feature
lgc_feat = gen_graph_feat(data, args)
# 根据feature相似度，计算节点间的联系
sim_edge = gen_edge_sim(data.edge_index, lgc_feat, data.edge_index.size(-1))
# 根据sim_edge构建亲和矩阵
adj_dic, adj_sim_dic = gen_adj_list(data.x.size(0), data.edge_index, sim_edge)
# 初始化边权重
row, col = data.edge_index
data.edge_attr = 1. / degree(col, data.num_nodes)[col]
# 如果是graph sampling，则根据对应的sampler建立loader
if args.sample=='GS':
    print("Sample method is GSGAE")
    # adj_dic, adj_dic_sim
    loader = AGSRandomWalkSampler(data, adj_dic= adj_dic, adj_sim_dic=adj_sim_dic, measure='uniform',
                                  alpha=1, batch_size=200, walk_length=4, num_steps=4,
                                  sample_coverage=200, save_dir=dataset.processed_dir, cal_norm=False)
else:
    print("To do in the future")


class Encoder(torch.nn.Module):
    # layer_dims包括初始feature维度，如cora[1433, 512, 512]等
    def __init__(self, layer_dims):
        super(Encoder, self).__init__()

        self.num_layers = len(layer_dims) - 1
        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(GCNConv(layer_dims[i], layer_dims[i+1], cached=True))


    # 调试：对隐层Z加入归一化，以保证其可以进行谱聚类
    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)

        out = x
        out = torch.relu(out)
        # out = self.scale(out)
        # out = F.normalize(out)
        return out

# 这里是DGI的腐蚀函数
def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index



channels = args.channels
layer_dims = [dataset.num_features, channels, channels]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = SAGES(feat_channels=dataset.num_features, hidden_channels=channels, encoder=Encoder(layer_dims),
              summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
              corruption=corruption).to(device)

x, train_pos_edge_index = data.x.to(device), data.edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(x, train_pos_edge_index)
    loss = model.loss(pos_z, neg_z, summary, train_pos_edge_index, x, 1, 0.0, 0.0)

    loss.backward()
    optimizer.step()



def test():
    model.eval()
    with torch.no_grad():
        z, _, _ = model(data.x.to(device), data.edge_index.to(device))

    node_embed = z.cpu().numpy()
    # 需要对node embedding进行特殊处理，去除掉均值
    node_embed = fill_ndarray(node_embed)

    classifier = Classifier(vectors=node_embed)
    acc, mic_f1s, macro_f1s = classifier(idx_train, idx_test, idx_val, data.y.numpy(), seed=0)
    # db, cluster_acc, cluster_nmi, cluster_adj = clustering(Cluster, node_embed, data.y.numpy())
    # return acc, mic_f1s, macro_f1s, db, cluster_acc, cluster_nmi, cluster_adj, node_embed
    return acc, mic_f1s, macro_f1s


best_acc, best_c_acc, best_c_nmi, best_c_adj = 0, 0, 0, 0
for epoch in range(1, 500):
    train()
    # acc, mic_f1s, macro_f1s, db, cluster_acc, cluster_nmi, cluster_adj, node_embed = test()
    # print('Epoch: {:03d}, acc: {:.4f}, mic_f1: {:.4f}， c_acc: {:.4f}, c_nmi: {:.4f}, c_adj: {:.4f}'.format(epoch, acc, mic_f1s, cluster_acc, cluster_nmi, cluster_adj))
    # best_acc, best_c_acc, best_c_nmi, best_c_adj = max(acc, best_acc), max(cluster_acc,best_c_acc), max(cluster_nmi, best_c_nmi), max(best_c_adj, cluster_adj)
    # 只进行节点分类
    acc, mic_f1s, macro_f1s = test()
    print('Epoch: {:03d}, acc: {:.4f}, mic_f1: {:.4f}'.format(epoch, acc, mic_f1s))
    best_acc = max(acc, best_acc)


# print("Best ACC: {:.4f}, Cluster_acc: {:.4f}, NMI: {:.4f}, Adj: {:.4f}".format(best_acc, best_c_acc, best_c_nmi, best_c_adj))
print("Best ACC: {:.4f}".format(best_acc))