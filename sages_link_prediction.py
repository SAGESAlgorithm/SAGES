# import os.path as osp
import argparse
import torch
import pickle
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import random

from datasets.planetoid import Planetoid
from gnn.sages_nn import SAGES

from torch_geometric.nn import GCNConv, GAE
from sklearn.metrics import roc_auc_score, average_precision_score

from util.classifier import Classifier
from util import utils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(19960812)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SAGES')
parser.add_argument('--dataset', type=str, default='CiteSeer')
parser.add_argument('--channels', type=int, default=1024)
parser.add_argument('--num_layers', type=int, default=5)
args = parser.parse_args()
assert args.model in ['GAE', 'VGAE', 'SAGES']
assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
kwargs = {'GAE': GAE, 'SAGES':SAGES}

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


class Encoder(torch.nn.Module):
    # layer_dims包括初始feature维度，如cora[1433, 512, 512]等
    def __init__(self, layer_dims):
        super(Encoder, self).__init__()

        self.num_layers = len(layer_dims) - 1
        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(GCNConv(layer_dims[i], layer_dims[i+1]))


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
        # out = self.scale(out)
        # out = F.normalize(out)
        return out

# 这里是DGI的腐蚀函数
def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


# 开始处理边的问题
# 这里的adj是添加自环的无向图
adj = utils.convert_undirec(utils.construct_adj(data))
# 这里需要将自环去掉
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
adj_orig = adj
# # mask_test+edges中也去掉了自环
# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = utils.mask_test_edges(adj)

# 处理边结束
with open('../data/citeseer_link_prediction.pkl','rb') as file:
    save_edges_dic = pickle.load(file)

coords, train_edges, val_edges = save_edges_dic['train_adj_edges'], save_edges_dic['trian_edges'], save_edges_dic['val_edges']
val_edges_false, test_edges, test_edges_false = save_edges_dic['val_edges_false'], save_edges_dic['test_edges'], save_edges_dic['test_edges_false']



channels = args.channels
layer_dims = [dataset.num_features, channels]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = SAGES(feat_channels=dataset.num_features, hidden_channels=channels, encoder=Encoder(layer_dims),
              summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
              corruption=corruption).to(device)

x, train_pos_edge_index = data.x.to(device), data.edge_index.to(device)
# 新加，测试edge_index
train_pos_edge_index = torch.from_numpy(coords).t().long().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def train():
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(x, train_pos_edge_index)

    loss = model.loss(pos_z, neg_z, summary, train_pos_edge_index, x, 1, 0.01, 0.01)

    loss.backward()
    optimizer.step()



def test():
    model.eval()
    with torch.no_grad():
        z, _, _ = model(data.x.to(device), train_pos_edge_index)

    node_embed = z.cpu().numpy()
    classifier = Classifier(vectors=node_embed)
    acc, mic_f1s, macro_f1s = classifier(idx_train, idx_test, idx_val, data.y.numpy(), seed=0)
    auc_score, ap_score = get_roc_score(node_embed, adj_orig, test_edges, test_edges_false)

    return acc, mic_f1s, macro_f1s, auc_score, ap_score


best_acc, best_auc, best_ap = 0, 0, 0
for epoch in range(1, 300):
    train()

    # 只进行节点分类&链接预测
    acc, mic_f1s, macro_f1s, auc_score, ap_score = test()
    print('Epoch: {:03d}, acc: {:.4f}, mic_f1: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format(epoch, acc, mic_f1s, auc_score, ap_score))
    if acc>best_acc:
        best_acc = acc
    if auc_score>best_auc:
        best_auc = auc_score
    if ap_score>best_ap:
        best_ap = ap_score


print("Best ACC: {:.4f}, auc: {:.4f}, ap: {:.4f}".format(best_acc, best_auc, best_ap))
