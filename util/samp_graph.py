import numpy as np
# graph sampling 简单低效版本，并行版本可以见AclusterAGAE/GSGAE_Gen_Graph部分

def softmax(z, temp):
    """
    temp用来控制各个数之间的比重，temp比较高，则相似度近的最后概率会变大
    :param z:
    :param temp:
    :return:
    """
    z = z / temp
    return np.exp(z) / sum(np.exp(z))

# 定义一种简单的游走一步的randomwalk算法，输入一系列当前节点，输出下一节点
def randomwalk_with_roots_one_step(roots, adj_dic, adj_sim_dic):
    """
    :param roots: root node id list,  numpy array
    :param adj_dic:     dic 邻接表
    :param adj_sim_dic: dic 邻接表（带权重）
    :return: next step nodes id list, numpy array
    """
    neigh_list = []
    for root in roots:
        neigh_nodes = adj_dic[root]
        neigh_proba = softmax(adj_sim_dic[root], temp=1.0)
        neigh_list.append(np.random.choice(a=neigh_nodes, size=1, replace=False, p=neigh_proba)[0])
    return np.unique(neigh_list)

def gen_subgraph(roots, walk_len, adj_dic, adj_sim_dic):
    """
    生成一步子图
    :param batch_size: 控制子图规模，即子图共有多少条rw
    :param step: 控制子图的深度，即每条random walk的长度
    :return: node_list: 返回node id list 即该子图所有的node, numpy array
    """
    nodes = roots
    for i in range(walk_len):
        nei_nodes = randomwalk_with_roots_one_step(roots, adj_dic, adj_sim_dic)
        nodes = np.hstack((nodes, nei_nodes))
        roots = nei_nodes
    return np.unique(nodes)