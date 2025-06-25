import torch
import torch.nn.functional as F
from networkx import non_edges

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from data import get_dataset
import os.path
import shutil
import numpy as np
from models import SimpleGCN
from getArgs import get_args
from otherEmbeding import getOtherEmbeding
from utils import *
from datasets import *
#  加载数据集
args = get_args()
args.name = args.dataset

print('has conv : '+str(args.conv)+'\n'
      +'has diffusion : '+str(args.diffusion)+'\n'
      +'has otherEmbeding : '+str(args.otherEmbeding)+'\n'
      +'the other type is : '+str(args.typeEmbeding)+'\n'
      +'the dataset is : '+str(args.dataset)+'\n'
      )
print('train:' + str(args.train_ratio) + '  val:' + str(args.val_ratio) + '  test:' + str((1-args.train_ratio-args.val_ratio)))
# 获取训练设备号
device = torch.device(f'{args.device}' if torch.cuda.is_available() else 'cpu')

# 获取数据和模型
# dataset = get_dataset(args,args.dataset,False).to(device)
# data = dataset[0].to(device)
dataset = get_dataset(args,args.dataset,False)
data = dataset.data.to(device)



heterophily = args.heterophily
edge_weight = None

file_max = 0
file_num = 1
if args.dataset == 'chameleon':
    with open('...../project/weight/DAEGC/pretrain' + '/max.txt', 'r', encoding='utf-8') as f:
        for line in f:
            file_num = file_num + 1
            file_max = max(file_max, float(line))

else:
    with open('...../project/weight/DAEGC/pretrain' + '/'+args.dataset+'_max.txt', 'r', encoding='utf-8') as f:
        for line in f:
            file_num = file_num + 1
            file_max = max(file_max, float(line))


if args.otherEmbeding:
    heterophily = getHeterophily(data)
    # 获取其他引入的嵌入表示
    otherEmbed = getOtherEmbeding(dataset,data,args).detach().to(device)
    # edge_weight = getOtherEmbeding(dataset, data, args).to(device)
    # if args.dataset == 'chameleon':
    #     with open('...../project/weight/DAEGC/pretrain' + '/max.txt', 'r', encoding='utf-8') as f:
    #         for line in f:
    #             file_num = file_num + 1
    #             file_max = max(file_max, float(line))
    #     otherEmbed = torch.load('...../project/weight/DAEGC/pretrain'+str(file_num-1)+'/embeddings.pth').to(device)
    # else:
    #     with open('...../project/weight/DAEGC/pretrain' + '/'+args.dataset+'_max.txt', 'r', encoding='utf-8') as f:
    #         for line in f:
    #             file_num = file_num + 1
    #             file_max = max(file_max, float(line))
    #     otherEmbed = torch.load(
    #         '...../project/weight/DAEGC/'+args.dataset+'pretrain' + str(file_num - 1) + '/'+args.dataset+'embeddings.pth').to(device)
else:
    otherEmbed = None
# chameleon的mask和out的维度不同，修改mask维度
dataset0 = Dataset01(name=args.dataset,
                          add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                          device=args.device,
                          use_sgc_features=args.use_sgc_features,
                          use_identity_features=args.use_identity_features,
                          use_adjacency_features=args.use_adjacency_features,
                          do_not_use_original_features=args.do_not_use_original_features)

model = SimpleGCN(args,otherEmbed,edge_weight,dataset,data,dataset0.num_classes,dataset0.num_node_features,heterophily).to(device)
if args.dataset in ['chameleon', 'squirrel', 'actor', 'sbm']:
    data.train_mask = dataset.train_mask.T[0]
    data.val_mask = dataset.val_mask.T[0]
    data.test_mask = dataset.test_mask.T[0]

# 自定义mask
if args.selfMask:
    split_by_label_flag = True
    if args.dataset in ['chameleon', 'cornell', 'texas']:
        split_by_label_flag = False
    idx_train, idx_val, idx_test =split_nodes(data.y.cpu(), args.train_ratio, args.val_ratio, (1-args.train_ratio-args.val_ratio), 15, split_by_label_flag)
    data.train_mask = idx_train
    data.val_mask = idx_val
    data.test_mask = idx_test

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 测试函数
def test():
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=1)
    y_true = data.y.cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_score = logits.exp().detach().cpu().numpy()

    # 计算F1分数
    f1_macro = f1_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()], average='macro')
    f1_micro = f1_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()], average='micro')

    # 计算acc
    acc = accuracy_score(y_true[data.test_mask.cpu().numpy()], y_pred[data.test_mask.cpu().numpy()])

    # 计算AUC
    try:
        auc = roc_auc_score(y_true[data.test_mask.cpu().numpy()], y_score[data.test_mask.cpu().numpy()], multi_class='ovr')
    except ValueError:
        auc = float('nan')

    return f1_macro, f1_micro, auc ,acc


# 训练和测试模型
best_f1_macro = 0
best_f1_micro = 0
best_auc = 0
best_acc = 0
best_f1_macro_epoch = 0
best_f1_micro_epoch = 0
best_auc_epoch = 0
best_acc_epoch = 0

for epoch in range(args.num_steps):
    train()
    f1_macro, f1_micro, auc, acc= test()

    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_f1_macro_epoch = epoch
    if f1_micro > best_f1_micro:
        best_f1_micro = f1_micro
        best_f1_micro_epoch = epoch
    if auc > best_auc and not torch.isnan(torch.tensor(auc)):
        best_auc = auc
        best_auc_epoch = epoch
    if acc > best_acc:
        best_acc = acc
        best_acc_epoch = epoch

    # if epoch % 10 == 0:
    #     print(f'Epoch {epoch}, F1-macro: {f1_macro:.4f}, F1-micro: {f1_micro:.4f}, AUC: {auc:.4f}')
print('run: '+str(args.run_num))
print(f'Best F1-macro: {best_f1_macro:.4f}'+'  best epoch : '+str(best_f1_macro_epoch))
print(f'Best F1-micro: {best_f1_micro:.4f}'+'  best epoch : '+str(best_f1_micro_epoch))
print(f'Best AUC: {best_auc:.4f}'+'  best epoch : '+str(best_auc_epoch))
print(f'Best ACC: {best_acc:.4f}'+'  best epoch : '+str(best_acc_epoch))
print('train:' + str(args.train_ratio) + '  val:' + str(args.val_ratio) + '  test:' + str((1-args.train_ratio-args.val_ratio)))

if args.dataset == 'chameleon' and args.otherEmbeding == True and args.diffusion == True:
    if not os.path.exists('...../project/weight/DAEGC/pretrain' + '/max.txt'):
        os.mknod('...../project/weight/DAEGC/pretrain' + '/max.txt', 0o666)

    if best_f1_macro >= file_max :
        with open('...../project/weight/DAEGC/pretrain' + '/max.txt', 'a', encoding='utf-8') as w:
            w.write(str(best_f1_macro)+ '\n')
        shutil.copytree('...../project/weight/DAEGC/pretrain','...../project/weight/DAEGC/pretrain'+str(file_num))
else:
    if best_f1_macro >= file_max and args.otherEmbeding == True and args.diffusion == True:
        with open('...../project/weight/DAEGC/pretrain' + '/'+args.dataset+'_max.txt', 'a', encoding='utf-8') as w:
            w.write(str(best_f1_macro)+ '\n')
        shutil.copytree('...../project/weight/DAEGC/pretrain','...../project/weight/DAEGC/'+args.dataset+'pretrain'+str(file_num))

print('has conv : '+str(args.conv)+'\n'
      +'has diffusion : '+str(args.diffusion)+'\n'
      +'has otherEmbeding : '+str(args.otherEmbeding)+'\n'
      +'the other type is : '+str(args.typeEmbeding)+'\n'
      +'the dataset is : '+str(args.dataset)+'\n'
      +'the heterophily is :'+str(heterophily)+'\n'
      )
result_file = '...../project/result_output/addEmbed_31'
file_name = str(args.dataset)
if args.otherEmbeding == True:
    file_name += '_'+str(args.typeEmbeding)
if args.conv == True:
    file_name += '_conv'
if args.diffusion == True:
    file_name += '_diffusion'

if not os.path.exists(result_file + '/result_'+file_name+'.txt'):
    os.mknod(result_file + '/result_'+file_name+'.txt', 0o666)
with open(result_file + '/result_'+file_name+'.txt', 'a', encoding='utf-8') as w:
    w.write('class-12-run:'+str(args.run_num) + '\n')
    w.write('BestF1-macro:'+str(best_f1_macro)+'\n')
    w.write('BestF1-micro:'+str(best_f1_micro)+'\n')
    w.write('BestAUC:'+str(best_auc)+'\n')
    w.write('BestACC:'+str(best_acc)+'\n')
    w.write('BestF1-macro_epoch:'+str(best_f1_macro_epoch)+'\n')
    w.write('BestF1-micro_epoch:'+str(best_f1_micro_epoch)+'\n')
    w.write('BestAUC_epoch:'+str(best_auc_epoch)+'\n')
    w.write('BestACC_epoch:'+str(best_acc_epoch)+'\n')
    w.write('Heterophily:'+str(heterophily)+'\n')
    w.write('noHeterophily:' + str(args.noheterophily) + '\n')
    w.write('train:' + str(args.train_ratio) + '  val:' + str(args.val_ratio) + '  test:' + str((1-args.train_ratio-args.val_ratio)) + '\n')
    w.write('alpha:' + str(args.alpha) + '  beta:' +str(args.beta)+ '  gamma:'+str(args.gamma)+'\n')



