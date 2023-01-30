# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:28:38 2022

"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import gudhi


def construct_subgraph(edge,adj_1,adj_2,adj_3,dis_mat,num_hop):
    row,col = edge[0], edge[1]
    row1,col1 = np.nonzero(adj_1[row,:]),np.nonzero(adj_1[col,:]) 
    row2,col2 = np.nonzero(adj_2[row,:]),np.nonzero(adj_2[col,:])
    row3,col3 = np.nonzero(adj_3[row,:]),np.nonzero(adj_3[col,:])
    
    if num_hop == 1:
        index_ = []
        index_.extend(row1[1])
        index_.extend(col1[1])
        index_.sort()    
        index_ = list(set(index_))
        tmp_adj = dis_mat[index_,]
        return_adj = tmp_adj[:,index_]
        return return_adj
    elif num_hop == 2:
        index_ = []
        index_.extend(row1[1])
        index_.extend(col1[1])
        index_.extend(row2[1])
        index_.extend(col2[1])
        index_.sort()     
        index_ = list(set(index_))
        tmp_adj = dis_mat[index_,]
        return_adj = tmp_adj[:,index_]
        return return_adj
    elif num_hop == 3:
        index_ = []
        index_.extend(row1[1])
        index_.extend(col1[1])
        index_.extend(row2[1])
        index_.extend(col2[1])
        index_.extend(row3[1])
        index_.extend(col3[1])
        index_.sort()
        index_ = list(set(index_))
        tmp_adj = dis_mat[index_,]
        return_adj = tmp_adj[:,index_]
        return return_adj
    
def graph_VETF(graph,num_H):
    
    data_ = []
    for i in range(len(graph)):
        if i != 0:
            data_.append(list(graph[i,0:i-1]))
    
    
    rips_complex = gudhi.RipsComplex(distance_matrix=data_, max_edge_length=1.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    N = []
    E = []
    T = []
    F = []
    for filtered_value in simplex_tree.get_filtration():
        if len(filtered_value[0]) == 1:
            N.append(filtered_value[0][0])
        elif len(filtered_value[0]) == 2:
            E.append(filtered_value[0])
        elif len(filtered_value[0]) == 3:
            T.append(filtered_value[0])
        elif len(filtered_value[0]) == 4:
            F.append(filtered_value[0])
    
    weight = [1] * len(N)
    return N,E,T,F,weight


def get_weighted_laplacian(V,E,T,F,weight):

    v_n = len(V)
    e_n = len(E)
    t_n = len(T)
    f_n = len(F)

    if e_n==0:
        return np.zeros((v_n,v_n)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0))
    
    # L0
    B1 = np.zeros((v_n,e_n))
    for j in range(e_n):
        one = E[j][0]
        two = E[j][1]
        index1 = V.index(one)
        index2 = V.index(two)
        B1[index1][j] = -weight[index2]
        B1[index2][j] = weight[index1]
    L0 = np.dot(B1,B1.T)
    
    if t_n==0:
        return L0,np.dot(B1.T,B1),np.zeros((0,0)),np.zeros((0,0))
    # L1
    B2 = np.zeros((e_n,t_n))
    for j in range(t_n):
        one = T[j][0]
        two = T[j][1]
        three = T[j][2]
        index1 = E.index([one,two])
        index2 = E.index([one,three])
        index3 = E.index([two,three])
        index4 = V.index(one)
        index5 = V.index(two)
        index6 = V.index(three)
        B2[index1][j] = weight[index6]
        B2[index2][j] = -weight[index5]
        B2[index3][j] = weight[index4]
    L1 = np.dot(B2,B2.T) + np.dot(B1.T,B1)


    if f_n==0:
        return L0,L1,np.dot(B2.T,B2),np.zeros((0,0))
    
    # L2
    B3 = np.zeros((t_n,f_n))
    for j in range(f_n):
        one = F[j][0]
        two = F[j][1]
        three = F[j][2]
        four = F[j][3]
        index1 = T.index([one,two,three])
        index11 = V.index(four)
        index2 = T.index([one,two,four])
        index22 = V.index(three)
        index3 = T.index([one,three,four])
        index33 = V.index(two)
        index4 = T.index([two,three,four])
        index44 = V.index(one)
        B3[index1][j] = -weight[index11]
        B3[index2][j] = weight[index22]
        B3[index3][j] = -weight[index33]
        B3[index4][j] = weight[index44]
    L2 = np.dot(B3,B3.T) + np.dot(B2.T,B2)
    #L3
    L3 = np.dot(B3.T,B3)
   
    return L0,L1,L2,L3

def get_value_for_a_L(L):
    zero = 0
    eigen0 = []
    value = np.linalg.eigvalsh(L) #计算特征值
    for v in value:
        if v>0.000000001:
            eigen0.append(round(v,8))
        else:
            zero = zero + 1
    temp0 = 0
    for v in eigen0:
        temp0 = temp0 + np.log(v)
    temp0 = -temp0
    
    return temp0,zero

def get_torsion(L0,L1,L2,L3):
    v0,zero0 = get_value_for_a_L(L0)
    v1,zero1 = get_value_for_a_L(L1)
    v2,zero2 = get_value_for_a_L(L2)
    v3,zero3 = get_value_for_a_L(L3)
    res = -0.5 * v1 + v2 - 1.5*v3
    return res,zero0,zero1,zero2

 
    

if __name__ == "__main__":

    positive = np.loadtxt("data/xxxxxx.edgelist", dtype=np.int64)    
    fw = open("data/xxxxxxx_torsion_1-1.edgelist","w")
    G = nx.Graph()
    G.add_edges_from(positive)
    unique_entity = len(G.nodes)
    
    adj = sp.coo_matrix((np.ones(positive.shape[0]), (positive[:, 0], positive[:, 1])),
                        shape=(unique_entity, unique_entity), dtype=np.float32)
    
    adj_1 = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_2 = adj.dot(adj_1)
    adj_2 = adj_2 + adj_2.T.multiply(adj_2.T > adj_2) - adj_2.multiply(adj_2.T > adj_2)
    adj_3 = adj_2.dot(adj_1)
    adj_3 = adj_3 + adj_3.T.multiply(adj_3.T > adj_3) - adj_3.multiply(adj_3.T > adj_3)

    adj_11 = adj_1.sign()
    adj_22 = adj_2.sign()
    adj_33 = adj_3.sign()
    
    adj_1 = adj_1.sign().toarray()
    adj_2 = adj_2.sign().toarray() * 2
    adj_3 = adj_3.sign().toarray() * 3
    dis_mat = np.zeros((len(adj_1),len(adj_1[0])))
        
    num_hop = 1
    num_H = [2,3]
    
    for i in range(len(adj_1)):
        for j in range(i+1,len(adj_1[i])):
            if num_hop == 1:
                dis_list = [adj_1[i,j]]
                if dis_list[0] == 0.0:
                    dis_mat[i,j] = 100
                    dis_mat[j,i] = 100                    
                else:
                    dis_mat[i,j] = dis_list[0]
                    dis_mat[j,i] = dis_list[0]    
                    
            elif num_hop == 2: 
                dis_list = [adj_1[i,j],adj_2[i,j]]
                dis_list.sort()
                if np.max(dis_list) == 0.0:
                    dis_mat[i,j] = 100
                    dis_mat[j,i] = 100                    
                else:
                    if dis_list[0] == 0.0:
                        dis_mat[i,j] = dis_list[1]
                        dis_mat[j,i] = dis_list[1]  
                    else:
                        dis_mat[i,j] = dis_list[0]
                        dis_mat[j,i] = dis_list[0]                          
                    
            elif num_hop == 3:
                dis_list = [adj_1[i,j],adj_2[i,j],adj_3[i,j]]
                dis_list.sort()
                if np.max(dis_list) == 0.0:
                    dis_mat[i,j] = 100
                    dis_mat[j,i] = 100
                elif np.max(dis_list) != 0.0:
                    if dis_list[1] == 0.0:
                        dis_mat[i,j] = dis_list[2]
                        dis_mat[j,i] = dis_list[2]
                    else:
                        if dis_list[0] == 0.0:
                            dis_mat[i,j] = dis_list[1]
                            dis_mat[j,i] = dis_list[1] 
                        else:
                            dis_mat[i,j] = dis_list[0]
                            dis_mat[j,i] = dis_list[0]                         
    
    for item in G.edges:
        sub_graph = construct_subgraph([item[0],item[1]],adj_11,adj_22,adj_33,dis_mat,num_hop)
        V,E,T,F,weight = graph_VETF(sub_graph,num_H)
        L0,L1,L2,L3 = get_weighted_laplacian(V,E,T,F, weight)
        
        torsion,z0,z1,z2 = get_torsion(L0,L1,L2,L3)
        fw.writelines(str(item[0])+"\t"+str(item[1])+"\t"+str(torsion))
        fw.writelines("\n")
























