import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import os

import utils
from model import GAT
from evaluation import eva


def pretrain(dataset,args,device):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.nc_alpha,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.nc_lr, weight_decay=args.nc_weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            # acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
        if epoch % 5 == 0:

            torch.save(
                model.state_dict(), f"...../project/weight/DAEGC/pretrain/predaegc_{args.name}_{epoch}.pkl"
            )


def getPreL(args):

    args.cuda = torch.cuda.is_available()
    device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')

    datasets = utils.get_datasetNew(args,args.name)
    dataset = datasets.data

    if args.name == "CiteSeer":
        args.nc_lr = 0.005
        args.k_d = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.nc_lr = 0.005
        args.k_d = None
        args.n_clusters = 7
    elif args.name == "PubMed":
        args.nc_lr = 0.001
        args.k_d = None
        args.n_clusters = 3
    elif args.name == "chameleon" or "chameleon-filtered":
        args.nc_lr = 0.001
        args.k_d = None
        args.n_clusters = 10
    elif args.name == "squirrel" or "squirrel-filtered":
        args.nc_lr = 0.001
        args.k_d = None
        args.n_clusters = 10
    elif args.name == "actor":
        args.nc_lr = 0.001
        args.k_d = None
        args.n_clusters = 5
    elif args.name == "sbm":
        args.nc_lr = 0.001
        args.k_d = None
        args.n_clusters = 5
    elif args.name == "amazon-ratings":
        args.nc_lr = 0.001
        args.k_d = None
        args.n_clusters = 5
    else:
        args.k_d = None

    args.input_dim = dataset.num_features

    pretrain(dataset,args,device)

