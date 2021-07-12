
import torch
import numpy as np
import scipy.sparse as sp
from preprocess.normalization import fetch_normalization, row_normalize
from time import perf_counter


def construct_adj(data):
    """
    :param data: pyG dataset
    :return:  sp.coo_matrix
    """
    N = data.x.size(0)
    sp_adj = sp.coo_matrix((np.ones((data.edge_index.size(1))),
                             (data.edge_index[0].numpy(), data.edge_index[1].numpy())), shape=(N, N))
    # may be undricted
    return sp_adj

def convert_undirec(adj):
    """
    given adj，return adj of undricted graph, and have self-loop
    :param adj: sparse adj
    :return:
    """
    adj = adj.tocoo()
    adj = adj + adj.T + sp.eye(adj.shape[0])
    adj.data = np.ones(len(adj.data))
    return adj

def prepare_graph_data(adj):
    """
    :param adj: saprse adj
    :return:
    """
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    #data =  adj.tocoo().data
    adj[adj > 0.0] = 1.0
    G_torch = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.tocoo()
    return G_torch, torch.LongTensor(adj.row), torch.LongTensor(adj.col)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def lgc_process_citation(adj, features, normalization="AugNormAdj"):
    """
    :param adj:         sparse adj, with self-loop and undrecited
    :param features:    numpy.array
    :param normalization:
    :return:
    """
    # features变为稀疏矩阵
    features = sp.csr_matrix(features)
    # 没看出用处，这里可能还是adj吧
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj是稀疏矩阵，且undirected以及有self-loop
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj, features

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def lgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    # 逻辑，首先收集所有边，然后打乱，然后选对应比例的边做测试以及验证集，然后删除上述测试边和验证边重构训练用的邻接矩阵
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    import pdb
    pdb.set_trace()

    # 返回上三角矩阵/下三角矩阵，这样的话，不会重复
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    #  adj_tuple[0].shape
    # (11596, 2)
    edges_all = sparse_to_tuple(adj)[0]
    #  sparse_to_tuple(adj)[0].shape
    # (23192, 2)
    # edges all是边a,b和b,a都包括
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    # 这里的edges_all的个数是edges的两倍，所以num_test和num_val都是一半，论文中写的是10%用来测试，5%用来验证集
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    #assert ~ismember(test_edges_false, edges_all)
    #assert ~ismember(val_edges_false, edges_all)
    #assert ~ismember(val_edges, train_edges)
    #assert ~ismember(test_edges, train_edges)
    #assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false