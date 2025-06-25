
from sklearn.preprocessing import normalize


import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor,HeterophilousGraphDataset,WikipediaNetwork, Actor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops

import torch

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import gather_csr, scatter
from sklearn.model_selection import train_test_split
import numpy as np
import os
import yaml
import scipy.sparse as sp
from datasets import *
from MyGraphDataset import *
#from memory_profiler import profile

DATA_PATH = '...../data'

def split_nodes(labels, train_ratio, val_ratio, test_ratio, random_state, split_by_label_flag):
    idx = torch.arange(labels.shape[0])
    if split_by_label_flag:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio, stratify=labels)
    else:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio)

    if val_ratio:
        labels_train_val = labels[idx_train]
        if split_by_label_flag:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio), stratify=labels_train_val)
        else:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio))
    else:
        idx_val = None

    return idx_train, idx_val, idx_test

def get_datasetNew(args,name: str, use_lcc: bool = True) -> InMemoryDataset:
    path = os.path.join(DATA_PATH, name)
    if name in ['Cora']:
        dataset = Planetoid(path, name)
    elif name in ['CiteSeer', 'PubMed']:
        dataset = Planetoid(DATA_PATH, name)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif name in ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(path, name)
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(DATA_PATH, name)
        return dataset
    elif name == 'actor':
        dataset = Actor(DATA_PATH + '/' + name)
        return dataset
    elif name in ['squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed','chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed']:
        dataset0 = Dataset01(name=args.dataset,
                          add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                          device=args.device,
                          use_sgc_features=args.use_sgc_features,
                          use_identity_features=args.use_identity_features,
                          use_adjacency_features=args.use_adjacency_features,
                          do_not_use_original_features=args.do_not_use_original_features)
        src, dst = dataset0.graph.edges()
        edge_index = torch.tensor(np.vstack([src.cpu().numpy(), dst.cpu().numpy()]), dtype=torch.long)
        x=torch.zeros(2223,dataset0.num_node_features)
        for feature_name, features in dataset0.graph.ndata.items():
            x = dataset0.graph.ndata[feature_name]
        x1 = x
        data = Data(
            x=x1,
            edge_index=edge_index,
            y=dataset0.labels,
            train_mask=dataset0.train_idx_list,
            test_mask=dataset0.test_idx_list,
            val_mask=dataset0.val_idx_list
        )
        dataset = MyGraphDataset(data,dataset0.num_classes,dataset0.num_node_features)
        dataset.data = data
        return dataset
    else:
        raise Exception('Unknown dataset.')


    return dataset


def get_datasetOld(dataset):
    datasets = Planetoid('./dataset', dataset)
    return datasets

def data_preprocessing(dataset):
    dataset.edge_index,_ =remove_self_loops(dataset.edge_index)
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


# 计算数据集的异配度
def getHeterophily(data):
    same_edges = 0
    total_edges = data.edge_index.size(1)

    # 获取节点特征
    node_features = data.x

    # 遍历每条边，判断特征是否相同
    for i in range(total_edges):
        node_a, node_b = data.edge_index[:, i]
        if data.y[node_a] == data.y[node_b]:
            same_edges += 1

    # 计算异配度
    heterophily = (1-(same_edges / total_edges if total_edges > 0 else 0))
    return heterophily


class Logger:
    def __init__(self, args, metric, num_data_splits):
        self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset=args.dataset, name=args.name)
        self.verbose = args.verbose
        self.metric = metric
        self.val_metrics = []
        self.test_metrics = []
        self.best_steps = []
        self.num_runs = args.num_runs
        self.num_data_splits = num_data_splits
        self.cur_run = None
        self.cur_data_split = None

        print(f'Results will be saved to {self.save_dir}.')
        with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

    def start_run(self, run, data_split):
        self.cur_run = run
        self.cur_data_split = data_split
        self.val_metrics.append(0)
        self.test_metrics.append(0)
        self.best_steps.append(None)

        if self.num_data_splits == 1:
            print(f'Starting run {run}/{self.num_runs}...')
        else:
            print(f'Starting run {run}/{self.num_runs} (using data split {data_split}/{self.num_data_splits})...')

    def update_metrics(self, metrics, step):
        if metrics[f'val {self.metric}'] > self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            self.test_metrics[-1] = metrics[f'test {self.metric}']
            self.best_steps[-1] = step

        if self.verbose:
            print(f'run: {self.cur_run:02d}, step: {step:03d}, '
                  f'train {self.metric}: {metrics[f"train {self.metric}"]:.4f}, '
                  f'val {self.metric}: {metrics[f"val {self.metric}"]:.4f}, '
                  f'test {self.metric}: {metrics[f"test {self.metric}"]:.4f}')

    def finish_run(self):
        self.save_metrics()
        print(f'Finished run {self.cur_run}. '
              f'Best val {self.metric}: {self.val_metrics[-1]:.4f}, '
              f'corresponding test {self.metric}: {self.test_metrics[-1]:.4f} '
              f'(step {self.best_steps[-1]}).\n')

    def save_metrics(self):
        num_runs = len(self.val_metrics)
        val_metric_mean = np.mean(self.val_metrics).item()
        val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan
        test_metric_mean = np.mean(self.test_metrics).item()
        test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan

        metrics = {
            'num runs': num_runs,
            f'val {self.metric} mean': val_metric_mean,
            f'val {self.metric} std': val_metric_std,
            f'test {self.metric} mean': test_metric_mean,
            f'test {self.metric} std': test_metric_std,
            f'val {self.metric} values': self.val_metrics,
            f'test {self.metric} values': self.test_metrics,
            'best steps': self.best_steps
        }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

    def print_metrics_summary(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        print(f'Finished {metrics["num runs"]} runs.')
        print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.4f}')
        print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.4f}')
        print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.4f}')
        print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.4f}')

    @staticmethod
    def get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir



def compute_D(a, b):
    t1 = a.unsqueeze(1).expand(len(a), len(a), a.shape[1])
    t2 = b.unsqueeze(0).expand(len(b), len(b), b.shape[1])
    d = (t1 - t2).pow(2).sum(2)
    return d

# def calculate_P(edge_index, x, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     absx = torch.norm(x, p=2, dim=1)
#     return ((torch.sum(x[row] * x[col], dim=1) / (absx[row] * absx[col])) * deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1) * (x[col] - x[row])
#
# def calculate_Q(edge_index, x, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     # absx = torch.norm(x, p=2, dim=1)
#     dx = x[col] - x[row]
#     absdx = torch.norm(dx, p=2, dim=1)
#     return ((torch.sum(dx * dx, dim=1) / (absdx + 1e-5)) * deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1) * (x[col] - x[row])
#

# 第一步用均值，第二部用s
def cal_g_gradient1(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]
    ones = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    deg = scatter_add(ones, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    gra = deg_inv[row].view(-1, 1) * (x[col] - x[row])
    avg_gra = scatter(gra, row, dim=-2, dim_size=x.size(0), reduce='add')

    # calculate similarity
    dx = x[row] - x[col]
    s = torch.norm(dx, p=2, dim=1)
    # sigma2 = torch.var(s)
    s = torch.exp(- (s * s) / (2 * sigma2 * sigma2))
    r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='add')
    coe = s.view(-1, 1) / (r[row] + 1e-12)
    result = scatter(avg_gra[row] * coe, col, dim=-2, dim_size=x.size(0), reduce='add')
    # result = scatter(avg_gra[row] * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), col, dim=-2, dim_size=x.size(0), reduce='sum')
    return result

# 第一步用ew，第二部用s+ew
def cal_g_gradient2(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    gra = deg_inv[row].view(-1, 1) * (x[col] - x[row])
    avg_gra = scatter(gra, row, dim=-2, dim_size=x.size(0), reduce='add')

    # calculate similarity
    dx = x[row] - x[col]
    s = torch.norm(dx, p=2, dim=1)
    s = (s * s) / (2 * sigma2 * sigma2)
    r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='add')
    coe = s.view(-1, 1) / (r[row] + 1e-6)
    result = scatter(avg_gra[row] * coe * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='sum')
    # result = scatter(avg_gra[row] * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), col, dim=-2, dim_size=x.size(0), reduce='sum')
    return result

#@profile(precision=4, stream=open('g2.log','w+'))
def cal_g_gradient2(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]

    onestep = scatter(x[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='sum')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='sum')

    return twostep

#@profile(precision=4, stream=open('g3.log','w+'))
def cal_g_gradient3(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]

    onestep = scatter((x[col] - x[row]) * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')

    # onestep = scatter((x[col] - x[row]) * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    # twostep = scatter(onestep[col] * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = feature_norm(twostep)
    return twostep

def cal_g_gradient6(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]

    onestep = scatter((x[col] - x[row]) * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    # twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')

    # onestep = scatter((x[col] - x[row]) * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    # twostep = scatter(onestep[col] * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    onestep = feature_norm(onestep)
    return onestep

def calAx(edge_index, x, edge_weight=None, sigma=0):
    row, col = edge_index[0], edge_index[1]

    d = x[col] - x[row]
    d2 = torch.sum(d * d, dim=1)

    coe = torch.exp(- d2 / 2) * (1 / (torch.sqrt(2 * 3.141592) * sigma))
    result = scatter(x[col] * coe, row, dim=-2,
                     dim_size=x.size(0), reduce='sum')
    return result


def cal_g_gradient4(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    ones = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(ones, col, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    onestep = scatter(deg_inv[row].view(-1, 1) * (x[col] - x[row]), row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = feature_norm(twostep)
    return twostep

# 正态分布计算系数
def cal_g_gradient5(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    gra = deg_inv[row].view(-1, 1) * (x[col] - x[row])
    avg_gra = scatter(gra, row, dim=-2, dim_size=x.size(0), reduce='add')
    abs_agra = torch.norm(avg_gra, p=2, dim=1)
    s = compute_D(x[row], x[col])
    s = (torch.sum(avg_gra[row] * avg_gra[col], dim=1) / (abs_agra[row] * abs_agra[col] + 1e-6))
    r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='add')
    coe = s.view(-1, 1) / (r[row] + 1e-6)
    result = scatter(avg_gra[row] * coe * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), col, dim=-2, dim_size=x.size(0), reduce='sum')

    return result

#@profile(precision=4, stream=open('ggat.log','w+'))
def cal_g_gradient_gat(edge_index, x, gat, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, dropout=0.1, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]
    avg_gra = scatter((x[col] - x[row]) * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    # result = gat(avg_gra, edge_index)
    # result = scatter(avg_gra[col] * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), row, dim=-2,dim_size=x.size(0), reduce='sum')
    result = gat(avg_gra, edge_index)
    return result

# def cal_Bx(edge_index, x, g, gamma,  edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     absx = torch.norm(x, p=2, dim=1)
#     s = torch.sum(x[row] * x[col], dim=1) / (absx[row] * absx[col] + 1e-6)
#     s = s * (deg_inv_sqrt[col] * deg_inv_sqrt[row])
#     r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='sum')
#     coe = s / (r[row] + 1e-6).view(-1)
#     # result = scatter((x[col] - x[row] - gamma * g[row]) * coe.view(-1,1), col, dim=-2, dim_size=g.size(0), reduce='sum')
#     result = scatter((x[col] - x[row] - gamma * g[row]) * coe.view(-1, 1), row, dim=-2, dim_size=g.size(0),
#                      reduce='sum')
#     return result
#
# def cal_Q(edge_index, x, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     dx = x[col] - x[row]
#     absdx = torch.norm(dx, p=2, dim=1)
#     return ((torch.sum(dx * dx, dim=1) / (absdx + 0.000001)) * deg_inv_sqrt[col] * deg_inv_sqrt[row])
#
#
#
# def calculate_PQ(edge_index, x, Q, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     absx = torch.norm(x, p=2, dim=1)
#     absdx = torch.norm(x[col] - x[row], p=2, dim=1)
#     return ((torch.sum(x[row] * x[col], dim=1) / (absx[row] * absx[col])) * deg_inv_sqrt[col]
#             * deg_inv_sqrt[row] / (absdx + 0.000001)).view(-1, 1) * (x[col] - x[row]) * Q


def read_config(args):
    # specify the model family

    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'prediction/config/{}/{}.yaml'.format(args.configfile, args.times))
    print(yamlPath)
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        # TODO
        config_dict = yaml.safe_load(cont)['g3'][args.dataset]

    if args.gpu == -1:
        device = torch.device('cpu')
    elif args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda', int(args.gpu))
        else:
            print("cuda is not available, please set 'gpu' -1")
    for key, value in config_dict.items():
        args.__setattr__(key, value)

    return args

def feature_norm(fea):
    device = fea.device
    epsilon = 1e-12
    fea_sum = torch.norm(fea, p=1, dim=1)
    fea_inv = 1 / np.maximum(fea_sum.detach().cpu().numpy(), epsilon)
    fea_inv = torch.from_numpy(fea_inv).to(device)
    fea_norm = fea * fea_inv.view(-1, 1)

    return fea_norm

def accuracy(output, label):
    """ Return accuracy of output compared to label.
    Parameters
    ----------
    output:
        output from model (torch.Tensor)
    label:
        node label (torch.Tensor)
    """
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return correct / len(label)


def sparse_mx_to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))

def prob_to_adj(mx, threshold):
    mx = np.triu(mx, 1)
    mx += mx.T
    (row, col) = np.where(mx > threshold)
    adj = sp.coo_matrix((np.ones(row.shape[0]), (row,col)), shape=(mx.shape[0], mx.shape[0]), dtype=np.int64)
    adj = sparse_mx_to_sparse_tensor(adj)
    return adj




def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'label_embeddings']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler
