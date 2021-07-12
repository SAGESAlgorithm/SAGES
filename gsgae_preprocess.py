import argparse
import torch
import numpy as np
from datasets.planetoid import Planetoid
from util import utils
from util.classifier import Classifier

# 该脚本用于根据SGC提前计算cora/pubmbed/citeseer， 可以指定adj的normalization方式，以及传播的深度

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                            'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--dataset', type=str, default="PubMed",
                    choices=['Cora', 'CiteSeer', 'PubMed'],
                    help='dataset to use.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--epochs', type=int, default=200,
                    help='epochs for node classification')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)


dataset = Planetoid(root='/home/wangjialin/pyG_example/data/'+args.dataset, name=args.dataset)

data = dataset[0]
device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

# 组织好labels, idx_train/val/test/labels 得到对应的tensor
def construct_idx(data):
    idx_train, idx_val, idx_test = [], [], []
    for i in range(data.x.size(0)):
        if data.train_mask[i]:
            idx_train.append(i)
        elif data.val_mask[i]:
            idx_val.append(i)
        elif data.test_mask[i]:
            idx_test.append(i)
    return idx_train, idx_val, idx_test

def gen_graph_feat(data, args):
    # 得到adj，先根据data构建无向图adj，且添加自环
    adj = utils.convert_undirec(utils.construct_adj(data))
    # 这里返回的adj和features都是tensor
    adj, features = utils.lgc_process_citation(adj=adj, features=data.x.numpy(), normalization=args.normalization)
    print("Finished data loading and processing.")

    processed_features, precompute_time = utils.lgc_precompute(features, adj, args.degree)

    # 利用classifier来训练得到结果, 测试node_embed效果
    classifier = Classifier(vectors=processed_features.cpu().numpy())
    idx_train, idx_test, idx_val = construct_idx(data)
    acc, mic_f1s, macro_f1s = classifier(idx_train, idx_test, idx_val, data.y.numpy(), seed=0)
    print("K is ",args.degree, " Dataset ",args.dataset," Acc is ", acc)
    return processed_features.cpu()


def gen_edge_sim(edge, sgc_feat, batch_size=1e6, cuda=False):
    """

    :param sgc_feat:    cpu tensor/ can be gpu tensor
    :param batch_size:  each batch edge num
    :return: sim_edge
    """
    E = edge.size(-1)
    row, col = edge[0], edge[1]
    sim_edge = torch.FloatTensor(E)
    len = row.size(-1)
    batch_num = (len // batch_size) + 1
    for i in range(batch_num):
        start_id = batch_size * i
        end_id = batch_size * (i + 1)
        if start_id > len: break
        if end_id > len: end_id = len

        batch_row_emb = sgc_feat[row[start_id:end_id]]
        batch_col_emb = sgc_feat[col[start_id:end_id]]
        batch_row_emb, batch_col_emb = batch_row_emb.to(device), batch_col_emb.to(device)
        batch_sim_edge = torch.cosine_similarity(batch_row_emb, batch_col_emb, dim=1)
        if cuda:
            sim_edge[start_id:end_id] = batch_sim_edge
        else:
            sim_edge[start_id:end_id] = batch_sim_edge.cpu()
    return sim_edge

def gen_adj_list(N, edge, sim_edge):
    """

    :param sim_edge: cpu tensor
    :return: dic: adj_dic, adj_sim_dic
    """
    adj_dic = {}
    adj_sim_dic = {}
    row, col, sim_edge = edge[0].numpy(), edge[1].numpy(), sim_edge.numpy()

    for edge_id in range(len(row)):
        # 由于
        if row[edge_id] not in adj_dic:
            adj_dic[row[edge_id]] = []
            adj_sim_dic[row[edge_id]] = []
        adj_dic[row[edge_id]].append(col[edge_id])
        adj_sim_dic[row[edge_id]].append(sim_edge[edge_id])

    # import pdb
    # pdb.set_trace()
    not_connected = 0
    for node_id in range(N):
        # citeseer中，会存在节点没有连接到其它节点中
        if node_id not in adj_dic:
            adj_dic[node_id] = [node_id]
            adj_sim_dic[node_id] = [1.]
            not_connected += 1
    print("In graph preprocess, the un-connected nodes number is ", not_connected)

    assert len(adj_dic)==N and len(adj_sim_dic)==N, "construct adj_dic failed!"

    # 这里sample邻居，没有包含自身，所以可能效果有问题？
    for node_id in range(N):
        # citeseer中，会存在节点没有连接到其它节点中
        adj_dic[node_id], idx = np.unique(adj_dic[node_id], return_index=True)
        adj_sim_dic[node_id] = np.array(adj_sim_dic[node_id])[idx]

    return adj_dic, adj_sim_dic

if __name__ == '__main__':

    for i in range(20):
        args.degree = i
        sgc_feat = gen_graph_feat(data, args)
    import pdb
    pdb.set_trace()

    sim_edge = gen_edge_sim(data.edge_index, sgc_feat, data.edge_index.size(-1))
    adj_dic, adj_sim_dic = gen_adj_list(data.x.size(0), data.edge_index, sim_edge)
    print("Hello world")














# use for gpu test
# import numpy as np
# from nn.sgc import SGC
# from util.convert_np_torch import np2tensor, tensor_cpu2gpu
# from util.unsup_pred import train_test_regression

# 定义gpu上的model
# idx_train, idx_val, idx_test = np2tensor([np.array(idx_train), np.array(idx_val), np.array(idx_test)])
# data_list = [adj, features, idx_train, idx_val, idx_test, labels]
# adj, features, idx_train, idx_val, idx_test, labels = tensor_cpu2gpu(data_list, device)
# model = SGC(features.size(1), labels.max().item() + 1)
# datalist = [train_features, test_features, labels[idx_train], labels[idx_test]]
# train_test_regression(model, datalist, args)

# 保存sgc_feat
# file = 'sgc_feat/'+ args.dataset+'-' + args.normalization + '_degree_' + str(args.degree) + '_' + args.dataset + '_feat.pt'
# torch.save([processed_features.cpu()], file)