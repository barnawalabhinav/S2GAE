import argparse
import os
import torch
from torch.utils.data import DataLoader
from ogb.graphproppred import Evaluator
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.nn import Node2Vec, global_mean_pool, global_add_pool, global_max_pool
import time
from torch_geometric.data import Data
from model import LPDecoder_ogb as LPDecoder
from model import GCN_mgaev3 as GCN
from model import SAGE_mgaev2 as SAGE
from model import GIN_mgaev2 as GIN
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score
from utils import edgemask_um, edgemask_dm, do_edge_split_nc
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score
import os.path as osp
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from model import GCN_mgaev3_withMLP
def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list

def train_SVM(X,Y,args):
    clf = svm.SVC(kernel='rbf',probability = True,class_weight='balanced',shrinking=False)
    is_labeled = Y == Y
    clf.fit(X[is_labeled],Y[is_labeled])
    return clf

def train_SVR(X,Y,args):
    clf = SVR(kernel='rbf')
    clf.fit(X,Y)
    return clf

def SVMmetrics(X,Y,SVM):
    is_labeled = Y == Y
    X = X[is_labeled]
    Y = Y[is_labeled]
    y_prob = SVM.predict_proba(X)[:, 1]
    y_pred = SVM.predict(X)
    roc_auc = 0
    try:
        roc_auc = roc_auc_score(Y,y_prob)
    except:
        roc_auc = 0
    acc = accuracy_score(Y,y_pred)
    prec = precision_score(Y,y_pred)
    return (roc_auc,acc,prec)

def NNClassMetrics(y_true,y_pred):
    accs = []
    aucs = []
    precs = []
    if (len(y_true.shape)==1):
        roc_auc = roc_auc_score(np.array(y_true.detach().cpu()),np.array(y_pred.detach().cpu()))
        yval = (y_pred>=0.5).float()
        acc = accuracy_score(np.array(y_true.detach().cpu()),np.array(yval.detach().cpu()))
        prec = precision_score(np.array(y_true.detach().cpu()),np.array(yval.detach().cpu()))
        return (roc_auc,acc,prec)

    tasks = y_true.shape[1]
    for i in range(tasks):
        is_labeled = y_true[:,i] == y_true[:,i]
        yt = y_true[:,i][is_labeled]
        yp = y_pred[:,i][is_labeled]
        ypb = (yp >= 0.5).float()
        roc_auc = 0
        try:
            roc_auc = roc_auc_score(np.array(yt.detach().cpu()),np.array(yp.detach().cpu()))
            aucs.append(roc_auc)
        except:
            roc_auc = 0

        acc = accuracy_score(np.array(yt.detach().cpu()),np.array(ypb.detach().cpu()))
        accs.append(acc)
        prec = 0
        try:
            prec = precision_score(np.array(yt.detach().cpu()),np.array(ypb.detach().cpu()))
            precs.append(prec)
        except:
            pass
    try:
        roc_auc = sum(aucs)/len(aucs)
    except:
        roc_auc = 0
    
    acc = sum(accs)/len(accs)
    try:
        prec = sum(precs)/len(precs)
    except:
        prec = 0
    return (roc_auc,acc,prec)

def NNRegMetrics(y_true,y_pred):
    return mean_squared_error(np.array(y_true.detach().cpu()),np.array(y_pred.detach().cpu()))

def SVRMetrics(X,Y,SVM):
    y_pred = SVM.predict(X)
    return mean_squared_error(Y,y_pred)

def labels_preds(model,loader,device):
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            preds.append(out.squeeze().cpu())
            labels.append(batch.y.squeeze().cpu())
        if (preds[-1].dim() == 0):
            preds[-1] = preds[-1].unsqueeze(0)
        if (labels[-1].dim() ==0):
            labels[-1] = labels[-1].unsqueeze(0)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
    return (labels,preds)


def print_regression_results(train_mse,valid_mse,test_mse,i,args):
    print()
    print("-----------------------------")
    print()
    print(args.dataset_config)
    print(args.pretrained_path)
    dic = dict()
    dic[0]='last'
    dic[1]='combine'
    if (args.freeze_backbone):
        print("Backbone Frozen!")
    else:
        print("Backbone not Frozen!")
    if (args.svm):
        print("Using SVM")
    else:
        print("Using MLP")
    print("Pooling used: ", args.pooling)
    print("Features: ",dic[i])
    print("Train RMSE: ",train_mse)
    print("Valid RMSE: ",valid_mse)
    print("Test RMSE: ",test_mse)
    print()
def print_results(train_auc,train_acc,valid_auc,valid_acc,test_auc,test_acc,train_prec,valid_prec,test_prec,i,args):
    print()
    print("-----------------------------")
    print()
    print(args.dataset_config)
    print(args.pretrained_path)
    dic = dict()
    dic[0]='last'
    dic[1]='combine'
    if (args.freeze_backbone):
        print("Backbone Frozen!")
    else:
        print("Backbone not Frozen!")
    if (args.svm):
        print("Using SVM")
    else:
        print("Using MLP")
    print("Pooling used: ", args.pooling)
    print("Features: ",dic[i])
    print("Train AUC: ", train_auc)
    print("Train ACC: ", train_acc)
    print("Train Precision: ",train_prec)
    print("Valid AUC: ", valid_auc)
    print("Valid ACC: ", valid_acc)
    print("Valid Precision: ",valid_prec)
    print("Test AUC: ", test_auc)
    print("Test ACC: ", test_acc)
    print("Test Precision: ",test_prec)
    print()

def main():
    parser = argparse.ArgumentParser(description='S2-GAE (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--dataset_config', type=str, default='./molecules_config/ogbg-molclintox.json')
    parser.add_argument('--use_valedges_as_input', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--decode_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--decode_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--graph_batch',type=float,default = 128)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--mask_type', type=str, default='dm',
                        help='dm | um')  # whether to use mask features
    parser.add_argument('--patience', type=int, default=50,
                        help='Use attribute or not')
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument('--pretrained_path',type=str,default='weight/s2gaesvm-GCN512_zinc250k_dm_2_hidd128-0.8-3-256_model140.pth')
    parser.add_argument('--pooling',type=str,default="add")
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help="Freeze the backbone (default: False)")
    parser.add_argument('--svm', action='store_true', default=False, help="Use SVM (default: False)")
    parser.add_argument('--scratch', action='store_true', default=False, help="Run from scratch (default: False)")
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    with open(args.dataset_config, 'r') as json_file:
        datasetJSON = json.load(json_file)

    path = osp.join('dataset/molecules')

    dataset = PygGraphPropPredDataset(name = datasetJSON['dataset']) 
    split_idx = dataset.get_idx_split() 

    if args.use_sage == 'SAGE':
        model = SAGE(9, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.use_sage == 'GIN':
        model = GIN(9, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(9, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    if (not args.scratch):
        model.load_state_dict(torch.load(args.pretrained_path))
        print('Pre-trained model loaded')
    else:
        print("Training From Scratch")
    
    print(args)
    ## Train SVN
    if (args.svm):
        if (args.pooling == "add"):
            pool_func = global_add_pool
        elif args.pooling == "mean":
            pool_func = global_mean_pool
        elif args.pooling == "max":
            pool_func = global_max_pool
        else:
            raise ValueError(args.pooling)
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=len(split_idx["train"]), shuffle=False)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=len(split_idx["valid"]), shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=len(split_idx["test"]), shuffle=False)
        train_data = None
        valid_data = None
        test_data = None
        for train_data_ in train_loader:
            train_data = train_data_
        for valid_data_ in valid_loader:
            valid_data = valid_data_
        for test_data_ in test_loader:
            test_data = test_data_
        train_data.to(device)
        valid_data.to(device)
        test_data.to(device)
        train_edge_index = train_data.edge_index
        valid_edge_index = valid_data.edge_index
        test_edge_index = test_data.edge_index
        train_data.full_adj_t = SparseTensor.from_edge_index(train_edge_index).t()
        valid_data.full_adj_t = SparseTensor.from_edge_index(valid_edge_index).t()
        test_data.full_adj_t = SparseTensor.from_edge_index(test_edge_index).t()
        train_labels = torch.cat([dataset[i].y for i in split_idx["train"]], dim=0)
        valid_labels = torch.cat([dataset[i].y for i in split_idx["valid"]], dim=0)
        test_labels = torch.cat([dataset[i].y for i in split_idx["test"]], dim=0)
        train_feature = model(train_data.x.float(),train_data.full_adj_t.float())
        train_feature = [pool_func(feature_,train_data.batch).detach() for feature_ in train_feature]
        train_feature_list = extract_feature_list_layer2(train_feature)
        
        valid_feature = model(valid_data.x.float(),valid_data.full_adj_t.float())
        valid_feature = [pool_func(feature_,valid_data.batch).detach() for feature_ in valid_feature]
        valid_feature_list = extract_feature_list_layer2(valid_feature)
        
        test_feature = model(test_data.x.float(),test_data.full_adj_t.float())
        test_feature = [pool_func(feature_,test_data.batch).detach() for feature_ in test_feature]
        test_feature_list = extract_feature_list_layer2(test_feature)
        number_of_tasks = datasetJSON['number_of_tasks']
        for i in range(len(train_feature_list)):
            if (i==1):
                break
            t1 = time.time()
            train_feat = train_feature_list[i]
            valid_feat = valid_feature_list[i]
            test_feat = test_feature_list[i]
            if (datasetJSON['task']=='Graph Classification'):
                train_aucs = []
                train_accs = []
                valid_aucs = []
                valid_accs = []
                test_aucs = []
                test_accs = []
                train_precs = []
                valid_precs = []
                test_precs = []
                for j in range(number_of_tasks):
                    print("Task number: ",j+1)
                    trained_SVM = train_SVM(train_feat.detach().cpu().numpy(),np.array(train_labels[:,j].detach().cpu()),args)
                    train_auc, train_acc,train_prec = SVMmetrics(train_feat.detach().cpu().numpy(),np.array(train_labels[:,j].detach().cpu()),trained_SVM)
                    valid_auc, valid_acc,valid_prec = SVMmetrics(valid_feat.detach().cpu().numpy(),np.array(valid_labels[:,j].detach().cpu()),trained_SVM)
                    test_auc, test_acc,test_prec = SVMmetrics(test_feat.detach().cpu().numpy(),np.array(test_labels[:,j].detach().cpu()),trained_SVM)
                    if (train_auc!=0 and valid_auc!=0 and test_auc!=0):
                        train_aucs.append(train_auc)
                        valid_aucs.append(valid_auc)
                        test_aucs.append(test_auc)
                    train_accs.append(train_acc)
                    valid_accs.append(valid_acc)
                    test_accs.append(test_acc)
                    train_precs.append(train_prec)
                    valid_precs.append(valid_prec)
                    test_precs.append(test_prec)
                    print_results(train_auc,train_acc,valid_auc,valid_acc,test_auc,test_acc,train_prec,valid_prec,test_prec,i,args)
                train_auc = sum(train_aucs)/len(train_aucs)
                valid_auc = sum(valid_aucs)/len(valid_aucs)
                test_auc = sum(test_aucs)/len(test_aucs)
                train_acc = sum(train_accs)/len(train_accs)
                valid_acc = sum(valid_accs)/len(valid_accs)
                test_acc = sum(test_accs)/len(test_accs)
                train_prec = sum(train_precs)/len(train_precs)
                valid_prec = sum(valid_precs)/len(valid_precs)
                test_prec = sum(test_precs)/len(test_precs)
                print("Averaged Results")
                print_results(train_auc,train_acc,valid_auc,valid_acc,test_auc,test_acc,train_prec,valid_prec,test_prec,i,args)
            else:
                trained_SVR = train_SVR(train_feat.detach().cpu().numpy(),np.array(train_labels[:,0].detach().cpu()),args)
                train_mse= SVRMetrics(train_feat.detach().cpu().numpy(),np.array(train_labels[:,0].detach().cpu()),trained_SVR)
                valid_mse= SVRMetrics(valid_feat.detach().cpu().numpy(),np.array(valid_labels[:,0].detach().cpu()),trained_SVR)
                test_mse= SVRMetrics(test_feat.detach().cpu().numpy(),np.array(test_labels[:,0].detach().cpu()),trained_SVR)
                print_regression_results(train_mse,valid_mse,test_mse,i,args)
            print("Time taken = ",time.time()-t1)
        ## Train MLP
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

        model_withMLP = GCN_mgaev3_withMLP(model,args.pooling,datasetJSON['number_of_tasks'],args.freeze_backbone)
        model_withMLP.to(device)
        optimizer = torch.optim.Adam(model_withMLP.parameters(), lr = 0.001,weight_decay = 1e-3)
        if (datasetJSON['task']=="Graph Classification"):
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.MSELoss()
        epochs = 200
        t = time.time()
        for epoch in range(epochs):
            if (time.time()-t>3000):
                break
            model_withMLP.train()
            total_loss = 0
            for data in train_loader:
                data.to(device)
                output = model_withMLP(data.x.float(),data.edge_index,data.batch)
                is_labeled = data.y == data.y
                if (len(is_labeled)==0):
                    continue
                loss = criterion(output[is_labeled].squeeze(), data.y[is_labeled].squeeze().float())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(total_loss)
            train_labels,train_preds = labels_preds(model_withMLP,train_loader,device)
            valid_labels,valid_preds = labels_preds(model_withMLP,valid_loader,device)
            test_labels,test_preds = labels_preds(model_withMLP,test_loader,device)

            if (datasetJSON['task']=="Graph Classification"):
                train_preds = torch.sigmoid(train_preds)
                valid_preds = torch.sigmoid(valid_preds)
                test_preds = torch.sigmoid(test_preds)
                train_auc,train_acc,train_prec = NNClassMetrics(train_labels,train_preds)
                valid_auc,valid_acc,valid_prec = NNClassMetrics(valid_labels,valid_preds)
                test_auc,test_acc,test_prec = NNClassMetrics(test_labels,test_preds)
                print("EPOCH: ", epoch)
                print_results(train_auc,train_acc,valid_auc,valid_acc,test_auc,test_acc,train_prec,valid_prec,test_prec,1,args)
            else:
                train_mse = NNRegMetrics(train_labels,train_preds)
                valid_mse = NNRegMetrics(valid_labels,valid_preds)
                test_mse = NNRegMetrics(test_labels,test_preds)
                print("EPOCH: ", epoch)
                print_regression_results(train_mse,valid_mse,test_mse,1,args)
        print("Time taken for 200 epochs: ", time.time()-t)


if __name__ == "__main__":
    main()
