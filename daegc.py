import argparse
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import utils
from model import GAT
from evaluation import eva


class DAEGC(nn.Module):
    def __init__(self,args, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def trainer(dataset,args,device):
    model = DAEGC(args=args ,num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.nc_alpha, num_clusters=args.n_clusters).to(device)

    optimizer = Adam(model.parameters(), lr=args.nc_lr, weight_decay=args.nc_weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    with torch.no_grad():
        _, z = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # eva(y, y_pred, 'pretrain')

    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(data, adj, M)
            
            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            # eva(y, q, epoch)

        A_pred, z, q = model(data, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return kmeans.fit_predict(z.data.cpu().numpy())

def getL(args):

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')
    datasets = utils.get_datasetNew(args,args.name)
    dataset = datasets.data

    if args.name == 'CiteSeer':
        args.nc_lr = 0.0001
        args.k_d = None
        args.n_clusters = 6
    elif args.name == 'Cora':
        args.nc_lr = 0.0001
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

    args.pretrain_path = f'...../project/weight/DAEGC/pretrain/predaegc_{args.name}_{args.epoch}.pkl'
    args.input_dim = dataset.num_features

    label = trainer(dataset,args,device)
    if args.name == "chameleon":
        torch.save(label,'...../project/weight/DAEGC/pretrain/label.pth')
    else:
        torch.save(label,'...../project/weight/DAEGC/pretrain/'+args.name+'label.pth')
    return label

