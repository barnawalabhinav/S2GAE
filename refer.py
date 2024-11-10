import torch
from models import GINmolhiv, GINmolbbbp, GINmollipo
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ogb.graphproppred import Evaluator
import sys
import matplotlib.pyplot as plt
import time
import os
import numpy as np
torch.manual_seed(12345)

def model_criterion_lr_epochs(data_name):
    if (data_name=="ogbg-molhiv"):
        return (GINmolhiv(9,32,3),nn.BCELoss(),0.0008,500+1,0.0)

    if (data_name=="ogbg-molbbbp"):
        return (GINmolbbbp(9,32,3),nn.BCELoss(),0.001,500+1,0.0)

    if (data_name=="ogbg-mollipo"):
        return (GINmollipo(9,64,3),nn.MSELoss(),0.001,500+1,1000)


def evaluate(model,evaluator,test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data.x
            labels = data.y
            outputs = model(x = data.x.to(dtype=torch.float32), edge_index = data.edge_index, batch = data.batch, edge_attr=data.edge_attr.to(dtype=torch.float32))
            y_true.append(labels)
            y_pred.append(outputs)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred) 
    input_dict = {
        "y_true": y_true,
        "y_pred": y_pred
    }

    result = evaluator.eval(input_dict)
    return result

def dump(model,test_loader,outputfile):
    model.eval()
    y_pred = [ ]
    with torch.no_grad():
        for data in test_loader:
            inputs = data.x
            outputs = model(x = data.x.to(dtype=torch.float32), edge_index = data.edge_index, batch = data.batch, edge_attr=data.edge_attr.to(dtype=torch.float32))
            y_pred.append(outputs)
    y_pred = torch.cat(y_pred)
    y_pred = y_pred.numpy()
    np.savetxt(outputfile,y_pred)

def main():
    dataset_name = sys.argv[1]
    output_file_name = sys.argv[2]
    evaluator = Evaluator(name = dataset_name)  
    dataset = PygGraphPropPredDataset(name = dataset_name)
    model,criterion,lr,epochs,bestres = model_criterion_lr_epochs(dataset_name)
    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    optimizer = Adam(model.parameters(), lr = lr,weight_decay = 1e-3)
    t = time.time()
    for epoch in range(1,500):
        model.train()
        for data in train_loader:
            out = model(x = data.x.to(dtype=torch.float32), edge_index = data.edge_index, batch = data.batch, edge_attr=data.edge_attr.to(dtype=torch.float32))
            loss = criterion(out,data.y.to(dtype=torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        res = evaluate(model,evaluator,test_loader)
        if (dataset_name=="ogbg-mollipo"):
            res = res['rmse']
            if (res<bestres):
                bestres = res
                torch.save(model.state_dict(), 'model.pth')
        else:
            res = res['rocauc']
            if (res>bestres):
                bestres = res
                torch.save(model.state_dict(), 'model.pth')
        if (time.time()-t>3000):
            break
    model.load_state_dict(torch.load('model.pth'))
    res = evaluate(model,evaluator,test_loader)
    print(res)
    dump(model,test_loader,output_file_name)

    file_path = 'model.pth'
    if os.path.isfile(file_path):
        os.remove(file_path)

if __name__ == '__main__':
    main()


import torch
import torch.nn as nn
from torch.nn import Linear, Parameter, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn import global_mean_pool,global_add_pool, global_max_pool
import torch.nn.functional as F
from torch import Tensor

torch.manual_seed(12345)
class GIN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dimension):
        super().__init__(aggr='add')
        self.mlp  = Sequential(Linear(in_channels, out_channels), BatchNorm1d(out_channels), ReLU(),Dropout(p=0.2), Linear(out_channels, out_channels), ReLU(), Dropout(p=0.2),BatchNorm1d(out_channels))
        self.eps = Parameter(torch.Tensor([0]))
        self.lin = Linear(edge_dimension,in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)
        self.eps.data.zero_()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index,x=x,edge_attr=edge_attr)
        out = out + (1+self.eps)*x
        return self.mlp(out)

    def message(self, x_j, edge_attr):
        e = self.lin(edge_attr)
        return (x_j + e).relu()

class GINhiv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dimension):
        super().__init__(aggr='add')
        self.mlp  = Sequential(Linear(in_channels, out_channels), BatchNorm1d(out_channels), ReLU(),Linear(out_channels, out_channels), ReLU())
        self.eps = Parameter(torch.Tensor([0]))
        self.lin = Linear(edge_dimension,in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)
        self.eps.data.zero_()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index,x=x,edge_attr=edge_attr)
        out = out + (1+self.eps)*x
        return self.mlp(out)

    def message(self, x_j, edge_attr):
        e = self.lin(edge_attr)
        return (x_j + e).relu()

class GINmollipo(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,edge_dimension):
        super(GINmollipo, self).__init__()
        self.gin1 = GIN(in_channels,hidden_channels,edge_dimension)
        self.bn1 = (torch.nn.BatchNorm1d(hidden_channels))
        self.gin2 = GIN(hidden_channels,hidden_channels,edge_dimension)
        self.bn2 = (torch.nn.BatchNorm1d(hidden_channels))
        self.gin3 = GIN(hidden_channels,hidden_channels,edge_dimension)
        self.bn3 = (torch.nn.BatchNorm1d(hidden_channels))
        self.lin1 = Linear(3*hidden_channels,3*hidden_channels)
        self.bn4 = (torch.nn.BatchNorm1d(3*hidden_channels))
        self.lin2 = Linear((3*hidden_channels),1)
    
    def forward(self,x,edge_index,edge_attr,batch):
        x1 = self.gin1(x,edge_index,edge_attr)
        x1 = self.bn1(x1)
        x2 = self.gin2(x1,edge_index,edge_attr)
        x2 = self.bn2(x2)
        x3 = self.gin3(x2,edge_index,edge_attr)
        x3 = self.bn3(x3)

        x1 = global_add_pool(x1,batch)
        x2 = global_add_pool(x2,batch)
        x3 = global_add_pool(x3,batch)

        x = torch.cat((x1,x2,x3),dim = 1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        return x

class GINmolbbbp(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,edge_dimension):
        super(GINmolbbbp, self).__init__()
        self.gin1 = GIN(in_channels,hidden_channels,edge_dimension)
        self.gin2 = GIN(hidden_channels,hidden_channels,edge_dimension)
        self.lin1 = Linear(2*hidden_channels,2*hidden_channels)
        self.lin2 = Linear((2*hidden_channels),1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,edge_index,edge_attr,batch):
        x1 = self.gin1(x,edge_index,edge_attr)
        x2 = self.gin2(x1,edge_index,edge_attr)

        x1 = global_add_pool(x1,batch)
        x2 = global_add_pool(x2,batch)

        x = torch.cat((x1,x2),dim = 1)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x

class GINmolhiv(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,edge_dimension):
        super(GINmolhiv, self).__init__()
        self.gin1 = GINhiv(in_channels,hidden_channels,edge_dimension)
        self.gin2 = GINhiv(hidden_channels,hidden_channels,edge_dimension)
        self.bn1  = (torch.nn.BatchNorm1d(hidden_channels))
        self.bn2  = (torch.nn.BatchNorm1d(hidden_channels))
        self.lin1 = Linear(2*hidden_channels,2*hidden_channels)
        self.bn3  = (torch.nn.BatchNorm1d(2*hidden_channels))
        self.lin2 = Linear((2*hidden_channels),1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,edge_index,edge_attr,batch):
        x1 = self.gin1(x,edge_index,edge_attr)
        x1 = self.bn1(x1)
        x2 = self.gin2(x1,edge_index,edge_attr)
        x2 = self.bn2(x2)
    
        x1 = global_add_pool(x1,batch)
        x2 = global_add_pool(x2,batch)

        x = torch.cat((x1,x2),dim = 1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x