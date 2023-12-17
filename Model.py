# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import warnings
from random import sample
import warnings
from  sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=Warning)



dataset=torch.load('D:\EEGresult\Graphdata\PreTrain')   

 
train_dataset = sample(dataset,864)
validation_dataset = sample(dataset,144)


class Delta_Net(torch.nn.Module):
 
    def __init__(self):
        super(Delta_Net, self).__init__()
        self.gat1 = GATConv(9, 45, heads=3) 
        self.gat2 = GATConv(3*45, 18, heads=3) 
        self.pooling = GATConv(3*18, 4)

    def forward(self, data): 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        x = self.gat1(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.pooling(x, edge_index)
        x = pyg_nn.global_max_pool(x, batch) 
        return F.log_softmax(x, dim=1) 

class Theta_Net(torch.nn.Module):

    def __init__(self):
        super(Theta_Net, self).__init__()
        self.gat1 = GATConv(9, 45, heads=3) 
        self.gat2 = GATConv(3*45, 18, heads=3) 
        self.pooling = GATConv(3*18, 4)
       
    def forward(self, data): 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        x = self.gat1(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.pooling(x, edge_index)
        x = pyg_nn.global_max_pool(x, batch) 
        return F.log_softmax(x, dim=1)
        
       
    
class Alpha_Net(torch.nn.Module):
 
    def __init__(self):
       super(Theta_Net, self).__init__()
       self.gat1 = GATConv(9, 45, heads=3) 
       self.gat2 = GATConv(3*45, 18, heads=3) 
       self.pooling = GATConv(3*18, 4)
     
    def forward(self, data): 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        x = self.gat1(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.pooling(x, edge_index)
        x = pyg_nn.global_max_pool(x, batch) 
        return F.log_softmax(x, dim=1)
   

class Beta_Net(torch.nn.Module):
 
    def __init__(self):
        super(Delta_Net, self).__init__()
        self.gat1 = GATConv(9, 45, heads=3) 
        self.gat2 = GATConv(3*45, 18, heads=3) 
        self.pooling = GATConv(3*18, 4)

    def forward(self, data): 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        x = self.gat1(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index) 
        x = F.relu(x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.pooling(x, edge_index)
        x = pyg_nn.global_max_pool(x, batch) 
        return F.log_softmax(x, dim=1) 

class Gamma_Net(torch.nn.Module):
 
    def __init__(self):
        super(Delta_Net, self).__init__()
        self.gat1 = GATConv(9, 45, heads=3) 
        self.gat2 = GATConv(3*45, 18, heads=3) 
        self.pooling = GATConv(3*18, 4)

    def forward(self, data): 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        print('raw data',x.size(),edge_index.size(),batch.size())
        x = self.gat1(x, edge_index) 
        print('First GAT data',x.size(),edge_index.size())
        x = F.relu(x) 
        x = F.dropout(x, p=0.3, training=self.training)
        print('After Dropout',x.size(),edge_index.size())
        x = self.gat2(x, edge_index) 
        print('Second GAT data',x.size(),edge_index.size())
        x = F.relu(x) 
        x = F.dropout(x, p=0.5, training=self.training)
        print('After Dropout',x.size(),edge_index.size())
        x = self.pooling(x, edge_index)
        x = pyg_nn.global_max_pool(x, batch) 
        print('After Pooling',x.size(),edge_index.size())
        return F.log_softmax(x, dim=1) 

 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_delta = Delta_Net() 
model_theta = Theta_Net() 
model_alpha = Alpha_Net() 
model_beta  = Beta_Net() 
model_gamma = Gamma_Net() 



optimizer_delta = torch.optim.Adam(model_delta.parameters(), lr=0.001) 
optimizer_theta = torch.optim.Adam(model_theta.parameters(), lr=0.001) 
optimizer_alpha = torch.optim.Adam(model_alpha.parameters(), lr=0.001) 
optimizer_beta  = torch.optim.Adam(model_beta.parameters(), lr=0.001)
optimizer_gamma = torch.optim.Adam(model_gamma.parameters(), lr=0.001)




loss_delta = torch.nn.CrossEntropyLoss()
loss_theta = torch.nn.CrossEntropyLoss()
loss_alpha = torch.nn.CrossEntropyLoss()
loss_beta = torch.nn.CrossEntropyLoss()
loss_gamma = torch.nn.CrossEntropyLoss()
loss_weight = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True) # 加载训练数据集，训练数据中分成每批次20个图片data数据


model_delta.train() 
model_theta.train()
model_alpha.train() 
model_beta.train()
model_gamma.train() 


for epoch in range(100): 
    loss_all = 0
    loss_all_delta = 0
    loss_all_theta = 0
    loss_all_alpha = 0
    loss_all_beta = 0
    loss_all_gamma = 0
    running_corrects_delta=0
    running_corrects_theta=0
    running_corrects_alpha=0
    running_corrects_beta=0
    running_corrects_gamma=0
    running_corrects_blend=0
         
    for data in train_loader: 
        
        optimizer_delta.zero_grad() 
        optimizer_theta.zero_grad() 
        optimizer_alpha.zero_grad() 
        optimizer_beta.zero_grad()
        optimizer_gamma.zero_grad() 
        
      
        data_delta=data['delta']
        data_theta=data['theta']
        data_alpha=data['alpha']
        data_beta=data['beta']
        data_gamma=data['gamma']
        
      
        output_delta = model_delta(data_delta) 
        output_theta = model_theta(data_theta) 
        output_alpha = model_alpha(data_alpha) 
        output_beta = model_beta(data_beta) 
        output_gamma = model_gamma(data_gamma) 
        #print('out_put_size',output_delta.size())

        
        label = data_alpha.y 
        
     
        _, pred_delta = torch.max(output_delta.data, 1)
        _, pred_theta = torch.max(output_theta.data, 1)
        _, pred_alpha = torch.max(output_alpha.data, 1)
        _, pred_beta  = torch.max(output_beta.data, 1)
        _, pred_gamma = torch.max(output_gamma.data, 1)

      
        loss_delta_temp = loss_delta(output_delta,label) 
        loss_theta_temp = loss_theta(output_theta,label) 
        loss_alpha_temp = loss_alpha(output_alpha,label) 
        loss_beta_temp = loss_beta(output_beta,label) 
        loss_gamma_temp = loss_gamma(output_gamma,label) 
        
       
        loss_delta_temp.backward() 
        loss_theta_temp.backward() 
        loss_alpha_temp.backward() 
        loss_beta_temp.backward()
        loss_gamma_temp.backward() 
        
    
        loss_all_delta += loss_delta_temp.item() 
        loss_all_theta += loss_theta_temp.item() 
        loss_all_alpha += loss_alpha_temp.item() 
        loss_all_beta += loss_beta_temp.item() 
        loss_all_gamma += loss_gamma_temp.item() 
        
      
        running_corrects_delta += torch.sum(pred_delta == label.data)
        running_corrects_theta += torch.sum(pred_theta == label.data)
        running_corrects_alpha += torch.sum(pred_alpha == label.data)
        running_corrects_beta += torch.sum(pred_beta == label.data)
        running_corrects_gamma += torch.sum(pred_gamma == label.data)
        
      
        weight_delta_temp=running_corrects_delta.numpy()
        weight_theta_temp=running_corrects_theta.numpy()
        weight_alpha_temp=running_corrects_alpha.numpy()
        weight_beta_temp=running_corrects_beta.numpy()
        weight_gamma_temp=running_corrects_gamma.numpy()
        
        weight_total=weight_delta_temp+weight_theta_temp+weight_alpha_temp+weight_beta_temp+weight_gamma_temp
        
        weight_delta_final=weight_delta_temp/weight_total
        weight_theta_final=weight_theta_temp/weight_total
        weight_alpha_final=weight_alpha_temp/weight_total
        weight_beta_final=weight_beta_temp/weight_total
        weight_gamma_final=weight_gamma_temp/weight_total
        

        output_blend = output_delta * weight_delta_final + output_theta * weight_theta_final + output_alpha * weight_alpha_final + output_beta * weight_beta_final + output_gamma * weight_gamma_final
        _, blend_pred = torch.max(output_blend.data, 1)
        running_corrects_blend += torch.sum(blend_pred == label.data)
        loss_all += loss_all_delta * weight_delta_final + loss_all_theta * weight_theta_final + loss_all_alpha * weight_alpha_final + loss_all_beta * weight_beta_final + loss_all_gamma * weight_gamma_final
        
        
        loss_weight_temp = loss_weight(output_blend,label)
        loss_weight_temp.backward()
        
    
        optimizer_delta.step()
        optimizer_theta.step()
        optimizer_alpha.step()
        optimizer_beta.step()
        optimizer_gamma.step()
    
    accuracy=accuracy_score(label, blend_pred)

    
    if epoch % 5 == 0:
        print(loss_all)  
        print('training accuracy',accuracy)
