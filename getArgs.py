import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nameOfmodel', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--diffusion',type=bool,  default=False, help='Use diffusion during training or not.')
    parser.add_argument('--conv',type=bool,  default=False, help='Use conv before diffiusion or not.')
    parser.add_argument('--run_num',type=int, default=0, help='Run number.')
    parser.add_argument('--selfMask', type=bool, default=False, help='Use self mask or not.')
    parser.add_argument('--otherEmbeding',type=bool, default=False, help='Use otherEmbeding during training or not.')
    parser.add_argument('--typeEmbeding', default='HGT', help=' the class of Embeding during training .', choices=['HGT','MetaPath2Vec'])
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='CiteSeer',
                        choices=['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions',
                                 'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
                                 'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
                                 'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin','Cora','CiteSeer','PubMed','sbm'])

    # model architecture
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep'])
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])

    # regularization
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.25)
    parser.add_argument('--val_ratio', type=float, default=0.25)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--hidden', type=int, default=64, help='hidden size')

    # 尝试2
    parser.add_argument('--alpha', type=float, default=0, help='tolerance to stop EM algorithm')
    parser.add_argument('--beta', type=float, default=1, help='tolerance to stop EM algorithm')
    parser.add_argument('--gamma', type=float, default=0, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma1', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma2', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--heterophily', type=float, default=0.5, help='ratio of otherEmbeding')
    parser.add_argument('--noheterophily', type=float, default=0.5, help='ratio of yuan Embeding')
    parser.add_argument('--k', type=int, default=10, help='diffusion time')


    # 聚类相关参数
    parser.add_argument('--name', type=str, default='CiteSeer')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--nc_lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--nc_weight_decay', type=int, default=5e-3)
    parser.add_argument('--nc_alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args