"""
model_v2.py implements the transformer model using torch, not dgl in a graph view, which can be more efficient.
"""
import torch as th
import numpy as np
from torch.nn import LayerNorm
import torch.nn.init as INIT
import copy
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import dgl
import math
from torch.nn import Linear
from transformer_torch import *

torch.set_default_tensor_type('torch.FloatTensor')


class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']

class Encoder_(nn.Module):
    def __init__(self, in_feat_gp, d_model, d_ff, d_k, d_v, n_heads, n_layers, dropout):
        '''
        :param in_feat_gp: the length of input features of pocket node
        :param d_model: Embedding Size
        :param d_ff: FeedForward dimension
        :param d_k: dimension of K(=Q)
        :param d_v: dimension of  V
        :param n_heads:  number of heads in Multi-Head Attention
        :param n_layers: number of Encoder  Layer
        :param dropout: dropout ratio
        '''
        super(Encoder_, self).__init__()
        self.in_feat_gp = in_feat_gp
        self.src_emb = nn.Linear(in_feat_gp, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, bg):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        ndata_x = bg.ndata['x'].view(bg.batch_size, -1, self.in_feat_gp)  # [batch_size, src_len, in_feat]
        ndata_pad = bg.ndata['pad'].view(bg.batch_size, -1)  # [batch_size, src_len]

        enc_outputs = self.src_emb(ndata_x)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(ndata_pad, ndata_pad)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        # return torch.sigmoid(h)
        return h


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class EdgeWeightAndSum_V2(nn.Module):
    """
    change the nn.Tanh() function to nn.Sigmoid()
    """

    def __init__(self, in_feats):
        super(EdgeWeightAndSum_V2, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, edge_feats):
        g.edata['e'] = edge_feats
        g.edata['w'] = self.atom_weighting(g.edata['e'])
        weights = g.edata['w']
        h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum, weights


class ITN_V2(nn.Module):
    '''
    treat the ligand molecule same as the pocket sequence using transformer (residue sequence)
    '''

    def __init__(self, in_feat_gp=75, d_model=200, d_ff=512, d_k=128, d_v=128, n_heads=4, n_layers=3,
                 dropout=0.20, glp_outdim=200, d_FC_layer=200, n_FC_layer=2, n_tasks=1):
        super(ITN_V2, self).__init__()
        self.d_model = d_model

        # transformer encoder layer for protein pocket sequence
        self.g_trans = Encoder_(in_feat_gp=in_feat_gp, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads,
                                n_layers=n_layers, dropout=dropout)

        # graph layer for ligand and protein residual interaction
        # self.glp_gnn = DTIConvGraph3Layer_V2(in_dim=d_model, out_dim=glp_outdim, dropout=dropout)
        self.glp_gnn = DTIConvGraph3Layer(in_dim=d_model + 1, out_dim=glp_outdim, dropout=dropout)

        # MLP predictor
        self.FC = FC(glp_outdim, d_FC_layer, n_FC_layer, dropout, n_tasks)

        # read out
        self.readout = EdgeWeightAndSum_V2(glp_outdim)

    def forward(self, bgl, bgp, bglp):
        # node representation calculation for the ligand residue sequence
        gl_residue_feats, g1_residue_self_attn = self.g_trans(bgl)  # [batch_size, src_len, d_model]
        gl_residue_feats = gl_residue_feats.view(-1, self.d_model)  # [batch_size*src_len, d_model]

        # node representation calculation for the pocket sequence
        gp_residue_feats, gp_residue_self_attn = self.g_trans(bgp)  # [batch_size, src_len, d_model]
        gp_residue_feats = gp_residue_feats.view(-1, self.d_model)  # [batch_size*src_len, d_model]

        # init the node features of ligand-pocket graph
        bgl_mask = bgl.ndata['pad'].view(-1, 1)  # [batch_size*src_len, 1]
        bgp_mask = bgp.ndata['pad'].view(-1, 1)  # [batch_size*src_len, 1]
        # mask the padding node
        glp_node_feats = gl_residue_feats * bgl_mask + gp_residue_feats * bgp_mask
        bglp_edge_feats = bglp.edata.pop('e')
        # edge update on the ligand-pocket graph
        glp_edge_feats3 = self.glp_gnn(bglp, glp_node_feats, bglp_edge_feats)

        readouts, edge_weights = self.readout(bglp, glp_edge_feats3)
        return self.FC(readouts), (g1_residue_self_attn, gp_residue_self_attn), edge_weights

