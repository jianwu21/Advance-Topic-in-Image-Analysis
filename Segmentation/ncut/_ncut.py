'''
Science caculate for normalized cut

Author: Jian Wu, IT and Cognition
University: University of Copenhagen
Contact: xcb479@alunmi.ku.dk
'''

import numpy as np
from scipy import sprse
import networks as nx


def DW_matrics(graph):
    '''
    Return the diagnonal and weight matrices of a graph
    '''
    W = nx.to_scipy_sparse_matrix(graph, format='csc')
    entries = W.sum(axis=0)
    D = sparse.dia_matrix((entries, 0), shape=W.shape).tocsc()

    return D, W


def ncut_cost(cut, D, W):
    cut = np.array(cut)


    assoc_a = D.data[cut].sum()
    assoc_b = D.data[~cut].sum()

    return (cut_cost / assoc_a) + (cut_cost / assoc_b)

