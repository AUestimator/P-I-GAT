# -*- coding: utf-8 -*-


import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
import numpy as np
import os
import math
from torch_geometric.data import DataLoader
import scipy.sparse as sp
from scipy import spatial
import sys



def read_data(file,dataset_dir):
    
	data = np.load(dataset_dir+"/"+file, allow_pickle=True) 
    
	return data



def extract_rhythm_feature(data):
    feature = data[0,0]
    
    delta_feature = feature[:,0:9] 
    theta_feature = feature[:,9:18]
    alpha_feature = feature[:,18:27]
    beta_feature  = feature[:,27:36]
    gamma_feature = feature[:,36:45]
    
    return delta_feature,theta_feature,alpha_feature,beta_feature,gamma_feature
 
 

def binary_pearson_edge(features):
    
    temp=np.empty((128,128))
    binary_matrix=np.empty((128,128))
    for i in range(0,128):  
        for j in range(0,128):
            feartue_1=features[i,:]
            feature_2=features[j,:]
            pearson_matrix = np.corrcoef(feartue_1,feature_2)
            pearson_value=pearson_matrix[0,1]
            pearson=abs(pearson_value)
            temp[i,j]=pearson
            if temp[i,j]>=0.80:
                binary_matrix[i,j]=1
            else:
                binary_matrix[i,j]=0
                
    return binary_matrix



def count_edges(binary_matrix,count_matrix):
    
    for i in range(0,128):  
        for j in range(0,128):
            count_matrix[i,j]+=binary_matrix[i,j]
            
    return count_matrix



def connect_edge(count_edge,N):
    edge_connect = np.empty((128,128))#预置
    probability  = count_edge/N
    for i in range(0,128):  
        for j in range(0,128):
            if probability[i,j]<=0.92:
                edge_connect[i,j] = 0
            else:
                edge_connect[i,j] = probability[i,j]
    return edge_connect




dataset_dir = "D:/EEGedge"

for file in os.listdir(dataset_dir):
    print("processing: ",file,"......")
    data=read_data(file,dataset_dir)
    #divide rhythm
    delta,theta,alpha,beta,gamma = extract_rhythm_feature(data)

    delta_edge = binary_pearson_edge(delta)
    theta_edge = binary_pearson_edge(theta)
    alpha_edge = binary_pearson_edge(alpha)
    beta_edge  = binary_pearson_edge(beta)
    gamma_edge = binary_pearson_edge(gamma)
    
    count_matrix=np.zeros([128,128])
    delta_edge_count = count_edges(delta_edge,count_matrix)
    theta_edge_count = count_edges(theta_edge,count_matrix)
    alpha_edge_count = count_edges(alpha_edge,count_matrix)
    beta_edge_count  = count_edges(beta_edge,count_matrix)
    gamma_edge_count = count_edges(gamma_edge,count_matrix)

    delta_BayesEdge = connect_edge(delta_edge_count)
    theta_BayesEdge = connect_edge(theta_edge_count)
    alpha_BayesEdge = connect_edge(alpha_edge_count)
    beta_BayesEdge  = connect_edge(beta_edge_count)
    gamma_BayesEdge = connect_edge(gamma_edge_count)
    
    delta_edge_save = sp.coo_matrix(delta_BayesEdge)
    theta_edge_save = sp.coo_matrix(theta_BayesEdge)
    alpha_edge_save = sp.coo_matrix(alpha_BayesEdge)
    beta_edge_save  = sp.coo_matrix(beta_BayesEdge)
    gamma_edge_save = sp.coo_matrix(gamma_BayesEdge)
    
torch.save(delta_edge_save,'D:/alpha_BayesEdge')
torch.save(theta_edge_save,'D:/beta_BayesEdge')
torch.save(alpha_edge_save,'D:/theta_BayesEdge')
torch.save(beta_edge_save, 'D:/beta_BayesEdge')
torch.save(gamma_edge_save,'D:/gamma_BayesEdge')
    
