# -*- coding: utf-8 -*-


import torch
from torch_geometric.data import Data
import numpy as np
import os
import scipy.sparse as sp


alpha_bayesedge=torch.load('D:/alpha_BayesEdge')
beta_bayesedge=torch.load('D:/beta_BayesEdge')
theta_bayesedge=torch.load('D:/theta_BayesEdge')
gamma_bayesedge=torch.load('D:/gamma_BayesEdge')
delta_bayesedge=torch.load('D:/delta_BayesEdge')

alpha_bayes_matrix=alpha_bayesedge.toarray() 
beta_bayes_matrix=beta_bayesedge.toarray() 
gamma_bayes_matrix=gamma_bayesedge.toarray() 
delta_bayes_matrix=delta_bayesedge.toarray() 
theta_bayes_matrix=theta_bayesedge.toarray() 



def read_data(file,dataset_dir):
	data = np.load(dataset_dir+"/"+file, allow_pickle=True) 
	return data

def count(feature):
    am1=np.empty((128,128))
    for i in range(0,128):  
        for j in range(0,128):
            a=feature[i,:]
            b=feature[j,:]
            pearson_matrix = np.corrcoef(a,b)
            pearson_value=pearson_matrix[0,1]
            am1[i,j]=pearson_value
    return am1
    
def SparseEdge(pearson_matrix):
    edge_temp = sp.coo_matrix(pearson_matrix)
    indices = np.vstack((edge_temp.row, edge_temp.col))
    edge_index = torch.LongTensor(indices)
    edge_attr=edge_temp.data
    return edge_index,edge_attr
    
    
def Transform_Graphdata(data):
    features=data[0,0] 

    label=data[1,0]
    label=[label]


    delta_feature=features[:,0:9]
    theta_feature=features[:,9:18]
    alpha_feature=features[:,18:27]
    beta_feature=features[:,27:36]
    gamma_feature=features[:,36:45]
 
    
    delta_temp_value=count(delta_feature)
    theta_temp_value=count(theta_feature)
    alpha_temp_value=count(alpha_feature)
    beta_temp_value=count(beta_feature)
    gamma_temp_value=count(gamma_feature)

    delta_value=delta_temp_value*delta_bayes_matrix
    theta_value=theta_temp_value*theta_bayes_matrix
    alpha_value=alpha_temp_value*alpha_bayes_matrix
    beta_value=beta_temp_value*beta_bayes_matrix
    gamma_value=gamma_temp_value*gamma_bayes_matrix
 
    delta_value = np.float16(delta_value)
    theta_value = np.float16(theta_value)
    alpha_value = np.float16(alpha_value)
    beta_value = np.float16(beta_value)
    gamma_value = np.float16(gamma_value)

    edge_index_delta,edge_attr_delta=SparseEdge(delta_value)
    edge_index_theta,edge_attr_theta=SparseEdge(theta_value)
    edge_index_alpha,edge_attr_alpha=SparseEdge(alpha_value)
    edge_index_beta,edge_attr_beta=SparseEdge(beta_value)
    edge_index_gamma,edge_attr_gamma=SparseEdge(gamma_value)


    x_delta = delta_feature
    x_delta = np.float16(x_delta)
    x_delta = torch.FloatTensor(x_delta)
  
    x_theta = theta_feature
    x_theta = np.float16(x_theta)
    x_theta = torch.FloatTensor(x_theta)
  
    x_alpha = alpha_feature
    x_alpha = np.float16(x_alpha)
    x_alpha = torch.FloatTensor(x_alpha)
   
    x_beta = beta_feature
    x_beta = np.float16(x_beta)
    x_beta = torch.FloatTensor(x_beta)

    x_gamma = gamma_feature
    x_gamma = np.float16(x_gamma)
    x_gamma = torch.FloatTensor(x_gamma)

    y = torch.LongTensor(label)

    graphdata_delta = Data(x=x_delta, edge_index=edge_index_delta, y=y,edge_attr=edge_attr_delta)
    graphdata_theta = Data(x=x_theta, edge_index=edge_index_theta, y=y,edge_attr=edge_attr_theta)
    graphdata_alpha = Data(x=x_alpha, edge_index=edge_index_alpha, y=y,edge_attr=edge_attr_alpha)
    graphdata_beta = Data(x=x_beta, edge_index=edge_index_beta, y=y,edge_attr=edge_attr_beta)
    graphdata_gamma = Data(x=x_gamma, edge_index=edge_index_gamma, y=y,edge_attr=edge_attr_gamma)
    dataset_graph=dict([('delta', graphdata_delta),('theta', graphdata_theta),('alpha', graphdata_alpha),('beta', graphdata_beta),('gamma', graphdata_gamma)])
    return dataset_graph

dataset_dir = "D:/EEG_tags"

dataset=[]
for file in os.listdir(dataset_dir):
    print("processing: ",file,"......")
    data=read_data(file,dataset_dir)
    graphdata=Transform_Graphdata(data)
    dataset.append(graphdata)
torch.save(dataset,'D:/EEGresult/Graphdata')