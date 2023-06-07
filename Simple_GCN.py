#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 05:38:13 2023

@author: hbonen
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

G = nx.Graph(name='G')

for i in range(6):
    G.add_node(i, name=i)

edges = [(0,1),(0,2),(1,2),(0,3),(3,4),(3,5),(4,5)]
G.add_edges_from(edges)

print('Graph Info:\n', nx.info(G))

print('\nGraph Nodes: ', G.nodes.data())

plt.figure(figsize=(80, 80))
pos = nx.kamada_kawai_layout(G)
options = {
    'node_color': 'red',  # first nsL colors from the Spectral palette
    'node_size': 200,
    'width': 0.5,
    'arrowstyle': '-',
    'arrowsize': 0.6,
}

nx.draw_networkx(G, pos=pos, with_labels = True, arrows=True, **options)

ax = plt.gca()
ax.collections[0].set_edgecolor("#000000")

plt.show()


A = np.array(nx.attr_matrix(G, node_attr='name')[0])
X = np.array(nx.attr_matrix(G, node_attr='name')[1])
X = np.expand_dims(X,axis=1)

print('Shape of A: ', A.shape)
print('\nShape of X: ', X.shape)
print('\nAdjacency Matrix (A):\n', A)
print('\nNode Features Matrix (X):\n', X)

AX = np.dot(A,X)
print("Dot product of A and X (AX):\n", AX)

G_self_loops = G.copy()

self_loops = []
for i in range(G.number_of_nodes()):
    self_loops.append((i,i))

G_self_loops.add_edges_from(self_loops)

print('Edges of G with self-loops:\n', G_self_loops.edges)

A_star = np.array(nx.attr_matrix(G_self_loops, node_attr='name')[0])
print('Adjacency Matrix of added self-loops G (A_star):\n', A_star)

AX = np.dot(A_star, X)
print('AX:\n', AX)

Deg_Mat = G_self_loops.degree()
print('Degree Matrix of added self-loops G (D): ', Deg_Mat)

D = np.diag([deg for (n,deg) in list(Deg_Mat)])
print('Degree Matrix of added self-loops G as numpy array (D):\n', D)

D_inv = np.linalg.inv(D)
print('Inverse of D:\n', D_inv)

DAX = np.dot(D_inv,AX)
print('DAX:\n', DAX)

D_half_norm = fractional_matrix_power(D, -0.5)
DADX = D_half_norm.dot(A_star).dot(D_half_norm).dot(X)
print('DADX:\n', DADX)

np.random.seed(77777)
n_h = 4 #number of neurons in the hidden layer
n_y = 2 #number of neurons in the output layer
W0 = np.random.randn(X.shape[1],n_h) * 0.01
W1 = np.random.randn(n_h,n_y) * 0.01

def relu(x):
    return np.maximum(0,x)

def gcn(A,H,W):
    I = np.identity(A.shape[0]) 
    A_star = A + I 
    D = np.diag(np.sum(A_star, axis=0)) 
    D_half_norm = fractional_matrix_power(D, -0.5) 
    eq = D_half_norm.dot(A_star).dot(D_half_norm).dot(H).dot(W)
    return relu(eq)

H1 = gcn(A,X,W0)
H2 = gcn(A,H1,W1)
print('Features Representation from GCN output:\n', H2)

def plot_features(H2):
    x = H2[:,0]
    y = H2[:,1]
    
    size = 40
    plt.figure()
    plt.scatter(x,y,size, c='red')
    plt.xlim([np.min(x)*0.9, np.max(x)*1.1])
    plt.ylim([-1, 1])
    plt.xlabel('Feature Representation Dimension 0')
    plt.ylabel('Feature Representation Dimension 1')
    plt.title('Feature Representation')

    for i,row in enumerate(H2):
        str = "{}".format(i)
        plt.annotate(str, (row[0],row[1]),fontsize=4, fontweight='bold')

    plt.show()


plot_features(H2)
