import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.base import DGLError
from torch.nn import Embedding, Sequential, Linear, ModuleList, Dropout
import dgl
import dgl.function as fn
from functools import partial
# import torch_scatter
import time

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x

def GlorotOrthogonal(tensor, scale=2.):
    torch.nn.init.orthogonal_(tensor)
    tensor.mul_(torch.sqrt(scale / ((tensor.size(0) + tensor.size(1)) * torch.var(tensor, unbiased=False))))

class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, bias=True):
        super(DenseLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        GlorotOrthogonal(self.fc.weight.data)
        if self.bias:
            self.fc.bias.data.zero_()
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_i = nn.Linear(n_h, n_h)
        # self.f_g = nn.Linear(n_h, n_h)
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embedding, grid_sizes, pos_samples, neg_samples, pos_bias=None, neg_bias=None):
        embedding = self.f_i(embedding)
        pos_embed = embedding[pos_samples]
        neg_emebd = embedding[neg_samples]
        grid_embed = dgl.ops.segment_reduce(grid_sizes, pos_embed, 'mean')
        # grid_embed = self.f_g(grid_embed)

        pos_grid_embed_list = []
        for i, k in enumerate(grid_sizes):
            pos_grid_embed_list += [grid_embed[i].repeat(k,1)]
        pos_grid_embed = th.cat(pos_grid_embed_list)

        grid_sizes_neg = grid_sizes * int(neg_samples.shape[0]/pos_samples.shape[0])
        neg_grid_embed_list = []
        for i, k in enumerate(grid_sizes_neg):
            neg_grid_embed_list += [grid_embed[i].repeat(k,1)]
        neg_grid_embed = th.cat(neg_grid_embed_list)

        pos_logits = th.squeeze(self.f_k(pos_embed, pos_grid_embed), 1)
        neg_logits = th.squeeze(self.f_k(neg_emebd, neg_grid_embed), 1)

        if pos_bias is not None:
            pos_logits += pos_bias
        if neg_bias is not None:
            neg_logits += neg_bias

        logits = th.cat((pos_logits, neg_logits))

        return logits

class TimeDiscriminator(nn.Module):
    def __init__(self, n_h):
        super(TimeDiscriminator, self).__init__()
        self.f_i = nn.Linear(n_h, n_h)
        # self.f_i_ = nn.Linear(n_h, n_h) # TODO: if needed
        # self.f_g = nn.Linear(n_h, n_h)
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embedding, embedding_, grid_sizes, pos_samples, neg_samples, pos_bias=None, neg_bias=None):
        # TODO: embedding or embedding_
        embedding_ = self.f_i(embedding_)
        pos_embed = embedding_[pos_samples]
        grid_embed = dgl.ops.segment_reduce(grid_sizes, pos_embed, 'mean')

        embedding = self.f_i(embedding)
        pos_embed = embedding[pos_samples]
        neg_emebd = embedding[neg_samples]
        
        # grid_embed = self.f_g(grid_embed)

        pos_grid_embed_list = []
        for i, k in enumerate(grid_sizes):
            pos_grid_embed_list += [grid_embed[i].repeat(k,1)]
        pos_grid_embed = th.cat(pos_grid_embed_list)

        grid_sizes_neg = grid_sizes * int(neg_samples.shape[0]/pos_samples.shape[0])
        neg_grid_embed_list = []
        for i, k in enumerate(grid_sizes_neg):
            neg_grid_embed_list += [grid_embed[i].repeat(k,1)]
        neg_grid_embed = th.cat(neg_grid_embed_list)

        pos_logits = th.squeeze(self.f_k(pos_embed, pos_grid_embed), 1)
        neg_logits = th.squeeze(self.f_k(neg_emebd, neg_grid_embed), 1)

        if pos_bias is not None:
            pos_logits += pos_bias
        if neg_bias is not None:
            neg_logits += neg_bias

        logits = th.cat((pos_logits, neg_logits))

        return logits

class DistRelConv(nn.Module):
    def __init__(self, in_feats, out_feats, dist_dim, num_neighbor, boundaries, dist_embed, feat_drop, activation=None, hop1_fc=False, elem_gate=False, merge='sum'):
        super(DistRelConv, self).__init__()
        self.merge = merge
        self.time_list = ['morning', 'midday', 'night', 'late-night']
        self.conv1 = HeteroGraphConv({
            '00': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate),
            '01': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate),
            '10': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate),
            '11': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate)},
            aggregate='sum') # if need? sum or mean or stack
        # self.conv1 = HeteroGraphConv({
        #     '00': dgl.nn.GraphConv(in_feats, out_feats, activation=activation, allow_zero_in_degree=True),
        #     '01': dgl.nn.GraphConv(in_feats, out_feats, activation=activation, allow_zero_in_degree=True),
        #     '10': dgl.nn.GraphConv(in_feats, out_feats, activation=activation, allow_zero_in_degree=True),
        #     '11': dgl.nn.GraphConv(in_feats, out_feats, activation=activation, allow_zero_in_degree=True)},
        #     aggregate='sum') # if need? sum or mean or stack
        
        # self.conv2 = dgl.nn.GraphConv(out_feats, out_feats, activation=activation, allow_zero_in_degree=True)
        # self.conv2 = DistGATLayer(out_feats, out_feats, dist_dim, boundaries, dist_embed, 0.2, 0.2, transform=True, activation=activation) # Need to check again when with dropout
        self.conv2 = DistPoolConv(out_feats, out_feats, dist_dim, boundaries, dist_embed, num_neighbor, 0., 0., transform=True, activation=activation) # Need to check again when with dropout
        # self.conv2 = GATLayer(out_feats, out_feats, 0., 0., transform=True, activation=activation) # Need to check again when with dropout
        # self.conv2 = DistAttn(out_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop=0., attn_drop=0., activation=activation)
        # self.conv2 = dgl.nn.GATConv(out_feats, out_feats, 1, feat_drop, feat_drop, activation=activation, allow_zero_in_degree=True)

    def forward(self, graph, feat):
        graph = graph.local_var()
        h_list = []
        for key in self.time_list:
            # etype_list = [e for e in graph.etypes if key in e and key != e] # night and late-night
            etype_list = [key+ids for ids in ['00', '01', '10', '11']]
            subg = graph.edge_type_subgraph(etype_list)
            h = self.conv1(subg, {'p':feat})
            h_list.append(h['p'])

        h_t_list = []
        for i, key in enumerate(self.time_list):
            # print(i)
            i1 = (i - 1) % 4
            i2 = (i + 1) % 4
            # etype_list = [e for e in graph.etypes if key in e]
            # etype_list += [e for e in graph.etypes if self.time_list[i1] in e]
            # etype_list += [e for e in graph.etypes if self.time_list[i2] in e]
            etype_list = [e for e in graph.etypes if key == e]
            etype_list += [e for e in graph.etypes if self.time_list[i1] == e]
            etype_list += [e for e in graph.etypes if self.time_list[i2] == e]
            h_t = torch.stack([h_list[i1], h_list[i], h_list[i2]],dim=1) # if need? two or three h
            # h_t = torch.stack([h_list[i1], h_list[i]],dim=1) 
            if self.merge == 'sum':
                h_t = torch.sum(h_t, dim=1)
            if self.merge == 'mean':
                h_t = torch.mean(h_t, dim=1)
            if self.merge == 'max':
                h_t = torch.max(h_t, dim=1)[0]

            # h_t = h_list[i]
            
            subg = dgl.to_homogeneous(graph.edge_type_subgraph(etype_list), ndata=['loc'])
            # subg = dgl.to_simple(subg.to('cpu')).to(0) # remove repeated edges()
            h_t = self.conv2(subg, h_t) # .squeeze()
            h_t_list.append(h_t)
        # h_t_list = torch.stack(h_t_list,dim=1).sum(1)
        # h_t_list = torch.stack(h_t_list,dim=1)
        return h_t_list

class TwoHopConv(nn.Module):
    def __init__(self, in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation=None, hop1_fc=False, elem_gate=False):
        super(TwoHopConv, self).__init__()
        self.hop1_fc = hop1_fc
        self.elem_gate = elem_gate
        self.boundaries = boundaries
        self.dist_embed = dist_embed
        self.fc2 = nn.Linear(in_feats, out_feats, bias=False)
        self.fcd = nn.Linear(dist_dim, out_feats, bias=False)
        if self.hop1_fc: # if need?
            self.fc1 = nn.Linear(in_feats, out_feats, bias=False)
        
        self.fc_w1 = nn.Linear(2 * out_feats, out_feats, bias=False)
        self.fc_w2 = nn.Linear(2 * out_feats, out_feats, bias=False)
        if not self.elem_gate: # if need?
            self.vec_a = nn.Linear(out_feats, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation # if need?
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu') # if need?
        if self.hop1_fc:
            nn.init.xavier_uniform_(self.fc1.weight) # or xavier_normal_
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fcd.weight)
        nn.init.xavier_uniform_(self.fc_w1.weight)
        nn.init.xavier_uniform_(self.fc_w2.weight)
        if not self.elem_gate:
            nn.init.xavier_uniform_(self.vec_a.weight)

    def cal_dist(self, edges):
        dist = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
        embed = self.fcd(self.dist_embed(torch.bucketize(dist, self.boundaries)))
        return {'dist': embed}

    def msg_function(self, edges):
        w1 = torch.cat([edges.dst['h2'], edges.data['dist']], dim=1)
        w2 = torch.cat([edges.data['h1'], edges.src['h2']], dim=1)
        # print(w1.shape, w2.shape)
        w1 = self.fc_w1(w1)
        w2 = self.fc_w2(w2)
        scores = w1 + w2
        if not self.elem_gate:
            scores = self.vec_a(scores)
        beta = self.sigmoid(scores)
        h = beta * edges.src['h2'] + edges.data['h1']
        h = edges.dst['d0'] * edges.src['d2'] * h
        return {'he': h}

    def forward(self, graph, feat):
        graph = graph.local_var()
        # dist = graph.edata['dist'] # if need to update dist embedding
        graph.apply_edges(self.cal_dist)
        feat = self.feat_drop(feat)
        # dist = self.feat_drop(feat) # if need?
        # hd = self.fcd(dist)
        h2 = self.fc2(feat)
        if self.hop1_fc:
            h1 = self.fc1(feat)
        else:
            h1 = h2

        # deg0 = graph.in_degrees().float().clamp(min=1)
        # deg0 = torch.pow(deg0, -0.5)
        # deg2 = graph.out_degrees().float().clamp(min=1) # if need?
        mid_ids = graph.edata['mid']
        graph.ndata['h2'] = h2
        graph.edata['h1'] = h1[mid_ids]
        # graph.edata['hd'] = hd
        graph.ndata['d0'] = torch.pow(graph.in_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.ndata['d2'] = torch.pow(graph.out_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.apply_edges(self.msg_function)

        graph.update_all(message_func=fn.copy_edge('he', 'm'), reduce_func=fn.sum(msg='m', out='x'))
        feat = graph.nodes['p'].data.pop('x')
        # if need fc_node_update? concat and update, or add self-loop and relu?
        if self.activation:
            feat = self.activation(feat)
        return feat

# class TwoHopConv(nn.Module):
#     def __init__(self, in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation=None, hop1_fc=False, elem_gate=False):
#         super(TwoHopConv, self).__init__()
#         self.hop1_fc = hop1_fc
#         self.elem_gate = elem_gate
#         self.boundaries = boundaries
#         self.dist_embed = dist_embed
#         self.fc2 = nn.Linear(in_feats, out_feats, bias=False)
#         self.fcd = nn.Linear(dist_dim, out_feats, bias=False)
#         if self.hop1_fc: # if need?
#             self.fc1 = nn.Linear(in_feats, out_feats, bias=False)
        
#         self.fc_w1 = nn.Linear(2 * out_feats, out_feats, bias=False)
#         self.fc_w2 = nn.Linear(2 * out_feats, out_feats, bias=False)
#         if not self.elem_gate: # if need?
#             self.vec_a = nn.Linear(out_feats, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.activation = activation # if need?
    
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu') # if need?
#         if self.hop1_fc:
#             nn.init.xavier_uniform_(self.fc1.weight) # or xavier_normal_
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fcd.weight)
#         nn.init.xavier_uniform_(self.fc_w1.weight)
#         nn.init.xavier_uniform_(self.fc_w2.weight)
#         if not self.elem_gate:
#             nn.init.xavier_uniform_(self.vec_a.weight)

#     def cal_dist(self, edges):
#         dist = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
#         embed = self.fcd(self.dist_embed(torch.bucketize(dist, self.boundaries)))
#         return {'dist': embed}

#     def msg_function(self, edges):
#         w1 = torch.cat([edges.dst['h2'], edges.data['dist']], dim=1)
#         w2 = torch.cat([edges.data['h1'], edges.src['h2']], dim=1)
#         # print(w1.shape, w2.shape)
#         w1 = self.fc_w1(w1)
#         w2 = self.fc_w2(w2)
#         scores = w1 + w2
#         if not self.elem_gate:
#             scores = self.vec_a(scores)
#         beta = self.sigmoid(scores)
#         h = beta * edges.src['h2'] + edges.data['h1']
#         h = edges.dst['d0'] * h
#         return {'he': h}

#     def forward(self, graph, feat):
#         graph = graph.local_var()
#         # dist = graph.edata['dist'] # if need to update dist embedding
#         graph.apply_edges(self.cal_dist)
#         feat = self.feat_drop(feat)
#         # dist = self.feat_drop(feat) # if need?
#         # hd = self.fcd(dist)
#         h2 = self.fc2(feat)
#         if self.hop1_fc:
#             h1 = self.fc1(feat)
#         else:
#             h1 = h2

#         # deg0 = graph.in_degrees().float().clamp(min=1)
#         # deg0 = torch.pow(deg0, -0.5)
#         # deg2 = graph.out_degrees().float().clamp(min=1) # if need?
#         mid_ids = graph.edata['mid']
#         graph.ndata['h2'] = h2
#         graph.edata['h1'] = h1[mid_ids]
#         # graph.edata['hd'] = hd
#         graph.ndata['d0'] = torch.pow(graph.in_degrees().float().clamp(min=1).view(-1, 1), -1)
#         graph.apply_edges(self.msg_function)

#         graph.update_all(message_func=fn.copy_edge('he', 'm'), reduce_func=fn.sum(msg='m', out='x'))
#         feat = graph.nodes['p'].data.pop('x')
#         # if need fc_node_update? concat and update, or add self-loop and relu?
#         if self.activation:
#             feat = self.activation(feat)
#         return feat


# class DistGATLayer(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  dist_dim,
#                  boundaries,
#                  dist_embed,
#                  feat_drop=0.,
#                  attn_drop=0.,
#                  residual=False,
#                  transform=False,
#                  activation=None):
#         super(DistGATLayer, self).__init__()
#         self.transform = transform
#         if self.transform:
#             self.fc = nn.Linear(in_feats, out_feats, bias=False)
#             attn_in_feats = 2 * out_feats
#         else:
#             attn_in_feats = 2 * in_feats
#         self.attn_fc = nn.Linear(out_feats * 2, out_feats, bias=True) # True; False
#         self.attn_out = nn.Linear(out_feats, 1, bias=False)

#         self.attn_fc1 = nn.Linear(2*out_feats, out_feats)
#         self.attn_fc2 = nn.Linear(2*out_feats, out_feats)
#         # self.attn_fc3 = nn.Linear(2*out_feats, out_feats)
#         # dist_dim = out_feats
        
#         # self.boundaries = torch.tensor([98., 201, 311, 442, 599, 793, 1065, 1496, 2269, 10000]).cuda()
#         self.boundaries = boundaries # torch.tensor([39., 67, 98, 132, 167, 201, 235, 272, 311, 351, 395, 442, 491, 542, 599, 659, 721, 793, 875, 965, 1065, 1180, 1323, 1496, 1703, 1951, 2269, 2659, 3441, 10000]).cuda()
#         self.embed = dist_embed # torch.nn.Embedding(len(self.boundaries) + 1, dist_dim)
#         self.G = nn.Linear(dist_dim, out_feats, bias=False)
#         # self.combine = Linear(out_feats * 3, out_feats)

#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.tanh = nn.Tanh() # nn.Tanh() # nn.Tanh(); nn.LeakyReLU(0.2)
#         if residual:
#             if in_feats != out_feats:
#                 self.res_fc = nn.Linear(
#                     in_feats, out_feats, bias=False)
#             else:
#                 self.res_fc = Identity()
#         else:
#             self.register_buffer('res_fc', None)
#         self.reset_parameters()
#         self.activation = activation

#     def reset_parameters(self):
#         """Reinitialize learnable parameters."""
#         gain = nn.init.calculate_gain('tanh') # tanh; relu
        
#         if self.transform:
#             nn.init.xavier_normal_(self.fc.weight)
#         nn.init.xavier_normal_(self.attn_out.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_fc1.weight)
#         nn.init.xavier_normal_(self.attn_fc2.weight)
#         nn.init.xavier_normal_(self.G.weight)
#         # nn.init.xavier_normal_(self.combine.weight, gain=gain)

#         if isinstance(self.res_fc, nn.Linear):
#             nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))
            
#     def combine_dist(self, edges):
#         dist = (edges.dst['pos'] - edges.src['pos']).norm(p=2, dim=1)
#         dist_embed = self.G(self.embed(torch.bucketize(dist, self.boundaries)))
#         h_dist = self.combine(dist_embed * edges.src['h'])
#         return {'dh': h_dist}
    
#     def combine_double_dist(self, edges):
#         dist1 = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
#         dist_ = (edges.src['loc'] - edges.data['inter_pos']).norm(p=2, dim=1)
#         dist1 = self.G(self.embed(torch.bucketize(dist1, self.boundaries)))
#         dist_ = self.G(self.embed(torch.bucketize(dist_, self.boundaries)))
#         # dist_embed1 = self.embed(torch.bucketize(dist1, self.boundaries))
#         # dist_embed2 = self.embed(torch.bucketize(dist2, self.boundaries))
#         # dist_embed_ = self.embed(torch.bucketize(dist_, self.boundaries))
#         # h_dist = torch.cat([edges.src['h'] * dist_embed1, edges.data['inter_h'] * dist_embed2, dist_embed_], dim=1)
#         # h_dist = torch.cat([edges.src['h'], dist_embed1], dim=1)
#         # h_dist = self.combine(h_dist)
#         # h_dist = edges.src['h'] + h_dist
#         # h_dist = self.combine(dist_embed * edges.src['h'])

#         # dist1 = self.attn_fc1(torch.cat([edges.src['h'], dist1], dim=1)) # v2
#         # dist_ = self.attn_fc2(torch.cat([edges.data['inter_h'], dist_], dim=1)) # v2
#         # return {'d0': dist1, 'd1':dist_} # v2
#         # return {'d0': dist1, 'd1':dist_, 'd2': edges.dst['h']} # v2catdst

#         dist1 = self.attn_fc1(torch.cat([edges.src['h'] * dist1, edges.data['inter_h'] * dist_], dim=1)) # v1
#         return {'d0': edges.dst['h'], 'd1':dist1} # v1
    
#     def building_inds_random(self, graph, num=1):
#         cuda = graph.device
#         graph = graph.to('cpu')
#         src, dst = graph.edges()
#         sg = dgl.sampling.sample_neighbors(graph, dst, num)
#         src_new = sg.edges()[0]
#         return src_new.to(cuda) 

#     def building_inds(self, graph):
#         src, dst = graph.edges()
#         max_ids = scatter(src, dst, reduce="max")
#         return max_ids[dst]

#     def edge_attention(self, edges):
#         # h_c = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
#         # h_c = torch.cat([edges.data['dh'], edges.dst['h']], dim=1)
#         # h_c = torch.cat([edges.data['d0'], edges.data['d1'], edges.data['d2']], dim=1)
#         h_c = torch.cat([edges.data['d0'], edges.data['d1']], dim=1)
#         # h_c = edges.data['d1'] + edges.data['d2'] + edges.data['d3']
#         # h_c = self.attn_fc(h_c)
#         h_c = self.attn_fc(h_c)
#         h_c = self.tanh(h_c)
#         h_s = self.attn_out(h_c)
#         # h_s = self.attn_out(h_c)
#         # h_s = self.tanh(h_s)
#         return {'e': h_s}
    
#     def message_func(self, edges):
#         # return {'h': edges.data['dh'], 'e': edges.data['e']}
#         return {'h': edges.src['h'], 'e': edges.data['e']}

#     def my_message_func(self, edges):
#         # return {'h': edges.data['dh'], 'e': edges.data['e']}
#         e = edges.data['a'] * edges.data['dh']
#         # e = edges.data['a'] * edges.src['h']
#         return {'m': e}

#     def reduce_func(self, nodes):
#         alpha = F.softmax(nodes.mailbox['e'], dim=1)
#         alpha = self.attn_drop(alpha)
#         z = torch.sum(alpha * nodes.mailbox['h'], dim=1)
#         return {'z': z}
    
#     def forward(self, graph, feat):
#         graph = graph.local_var()
#         feat = self.feat_drop(feat)
#         if self.transform:
#             feat = self.fc(feat)

#         graph.ndata['h'] = feat
#         locations = graph.ndata['loc'] # .squeeze()

#         inter_ids = self.building_inds_random(graph)
#         graph.edata['inter_pos'] = locations[inter_ids]
#         graph.edata['inter_h'] = feat[inter_ids]

#         graph.apply_edges(self.combine_double_dist)
#         graph.apply_edges(self.edge_attention)
#         # graph.update_all(self.message_func, self.reduce_func)
#         # rst = graph.ndata.pop('h')

#         e = graph.edata.pop('e')
#         # compute softmax
#         graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
#         # message passing
#         # graph.apply_edges(self.my_message_func)
#         graph.update_all(fn.u_mul_e('h', 'a', 'm'),
#                             fn.sum('m', 'ft'))
#         rst = graph.dstdata['ft']

#         # residual
#         if self.res_fc is not None:
#             resval = self.res_fc(feat).view(feat.shape[0], -1, self._out_feats)
#             rst = rst + resval
#         # activation
#         if self.activation:
#             rst = self.activation(rst)
#         return rst

# v3
class DistGATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dist_dim,
                 boundaries,
                 dist_embed,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=False,
                 transform=False,
                 activation=None):
        super(DistGATLayer, self).__init__()
        self.transform = transform
        if self.transform:
            self.fc = nn.Linear(in_feats, out_feats, bias=False)
            attn_in_feats = 2 * out_feats
        else:
            attn_in_feats = 2 * in_feats
        self.attn_fc = nn.Linear(out_feats * 2, out_feats, bias=True) # True; False
        # self.attn_out = nn.Linear(out_feats, 1, bias=False)

        # self.attn_fc1 = nn.Linear(2*out_feats, out_feats)
        # self.attn_fc2 = nn.Linear(2*out_feats, out_feats)
        # self.attn_fc3 = nn.Linear(2*out_feats, out_feats)
        # dist_dim = out_feats
        
        # self.boundaries = torch.tensor([98., 201, 311, 442, 599, 793, 1065, 1496, 2269, 10000]).cuda()
        self.boundaries = boundaries # torch.tensor([39., 67, 98, 132, 167, 201, 235, 272, 311, 351, 395, 442, 491, 542, 599, 659, 721, 793, 875, 965, 1065, 1180, 1323, 1496, 1703, 1951, 2269, 2659, 3441, 10000]).cuda()
        self.embed = dist_embed # torch.nn.Embedding(len(self.boundaries) + 1, dist_dim)
        self.G = nn.Linear(dist_dim, out_feats, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.tanh = nn.Tanh() # nn.Tanh(); n.LeakyReLU(0.2)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(
                    in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh') # tanh; relu
        
        if self.transform:
            nn.init.xavier_normal_(self.fc.weight)
        # nn.init.xavier_normal_(self.attn_out.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight)

        nn.init.xavier_normal_(self.G.weight)
        # nn.init.xavier_normal_(self.combine.weight, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))
            
    def combine_dist(self, edges):
        dist = (edges.dst['pos'] - edges.src['pos']).norm(p=2, dim=1)
        dist_embed = self.G(self.embed(torch.bucketize(dist, self.boundaries)))
        h_dist = self.combine(dist_embed * edges.src['h'])
        return {'dh': h_dist}
    
    def combine_double_dist(self, edges):
        dist1 = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
        # dist2 = (edges.dst['loc'] - edges.data['inter_pos']).norm(p=2, dim=1)
        dist_ = (edges.src['loc'] - edges.data['inter_pos']).norm(p=2, dim=1)
        dist_embed1 = self.G(self.embed(torch.bucketize(dist1, self.boundaries)))
        # dist_embed2 = self.G(self.embed(torch.bucketize(dist2, self.boundaries)))
        dist_embed_ = self.G(self.embed(torch.bucketize(dist_, self.boundaries)))
        # dist_embed1 = self.embed(torch.bucketize(dist1, self.boundaries))
        # dist_embed2 = self.embed(torch.bucketize(dist2, self.boundaries))
        # dist_embed_ = self.embed(torch.bucketize(dist_, self.boundaries))
        # h_dist = torch.cat([edges.src['h'] * dist_embed1, edges.data['inter_h'] * dist_embed2, dist_embed_], dim=1)
        # h_dist = torch.cat([edges.src['h'], dist_embed1], dim=1)
        # h_dist = self.combine(h_dist)
        # h_dist = edges.src['h'] + h_dist
        h_dist = edges.dst['d0'] * edges.src['d2'] * edges.src['h'] # v0
        # h_dist = edges.dst['d0'] * edges.src['d2'] * self.attn_fc(dist_embed1 * edges.src['h']) # v1
        # h_dist = edges.dst['d0'] * edges.src['d2'] * self.attn_fc(dist_embed1 * edges.src['h'] + dist_embed_ * edges.data['inter_h'] ) # v2
        # h_dist = edges.dst['d0'] * edges.src['d2'] * self.attn_fc(torch.cat([dist_embed1 * edges.src['h'], dist_embed_ * edges.data['inter_h']], dim=1)) # v3
        # h_dist = edges.dst['d0'] * edges.src['d2'] * (edges.src['h'] + self.attn_fc(torch.cat([dist_embed1, dist_embed_, edges.data['inter_h']], dim=1))) # v5, gated/pooling/new weight
        return {'h': h_dist}
        # d1 = self.attn_fc1(torch.cat([edges.src['h'], dist_embed1], dim=1))
        # d2 = self.attn_fc2(torch.cat([edges.data['inter_h'], dist_embed2], dim=1))
        # d3 = self.attn_fc3(torch.cat([edges.dst['h'], dist_embed_], dim=1))
        # return {'d1':d1, 'd2':d2, 'd3':d3}
    
    def building_inds_random(self, graph, num=1):
        cuda = graph.device
        graph = graph.to('cpu')
        src, dst = graph.edges()
        sg = dgl.sampling.sample_neighbors(graph, dst, num)
        src_new = sg.edges()[0]
        return src_new.to(cuda)

    def building_inds(self, graph):
        src, dst = graph.edges()
        max_ids = scatter(src, dst, reduce="max")
        return max_ids[dst]

    def edge_attention(self, edges):
        # h_c = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        # h_c = torch.cat([edges.data['dh'], edges.dst['h']], dim=1)
        h_c = torch.cat([edges.data['d1'],edges.data['d2'],edges.data['d3']], dim=1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {'e': h_s}
    
    def message_func(self, edges):
        # return {'h': edges.data['dh'], 'e': edges.data['e']}
        return {'h': edges.src['h'], 'e': edges.data['e']}

    def my_message_func(self, edges):
        # return {'h': edges.data['dh'], 'e': edges.data['e']}
        e = edges.data['a'] * edges.data['dh']
        # e = edges.data['a'] * edges.src['h']
        return {'m': e}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.attn_drop(alpha)
        z = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'z': z}
    
    def forward(self, graph, feat):
        graph = graph.local_var()
        # feat = self.feat_drop(feat)
        if self.transform:
            feat = self.fc(feat)

        graph.ndata['h'] = feat
        locations = graph.ndata['loc']

        inter_ids = self.building_inds_random(graph)
        graph.edata['inter_pos'] = locations[inter_ids]
        graph.edata['inter_h'] = feat[inter_ids]
        
        graph.ndata['d0'] = torch.pow(graph.in_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.ndata['d2'] = torch.pow(graph.out_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.apply_edges(self.combine_double_dist)
        # graph.apply_edges(self.edge_attention)
        # graph.update_all(self.message_func, self.reduce_func)
        # rst = graph.ndata.pop('h')

        # e = graph.edata.pop('e')
        # compute softmax
        # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        # graph.apply_edges(self.my_message_func)
        graph.update_all(fn.copy_e('h', 'm'),
                            fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(feat).view(feat.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

# v6
class DistPoolConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dist_dim,
                 boundaries,
                 dist_embed,
                 num_neighbor=5,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=False,
                 transform=False,
                 activation=None):
        super(DistPoolConv, self).__init__()
        self.num = num_neighbor
        self.out_feats = out_feats
        self.dist_dim = dist_dim
        self.transform = transform
        if self.transform:
            self.fc = nn.Linear(in_feats, out_feats, bias=False)
            attn_in_feats = 2 * out_feats
        else:
            attn_in_feats = 2 * in_feats
        self.agg_fc = nn.Linear(out_feats * 2, out_feats, bias=True) # True; False
        # self.attn_fc = nn.Linear(out_feats * 1, out_feats, bias=True) # True; False
        # self.attn_fc = nn.Parameter(torch.FloatTensor(size=(3, out_feats, out_feats)))
        # self.attn_out = nn.Linear(out_feats, 1, bias=False)

        # self.attn_fc1 = nn.Linear(2*out_feats, out_feats)
        # self.attn_fc2 = nn.Linear(2*out_feats, out_feats)
        # self.attn_fc3 = nn.Linear(2*out_feats, out_feats)
        # dist_dim = out_feats
        
        # self.boundaries = torch.tensor([98., 201, 311, 442, 599, 793, 1065, 1496, 2269, 10000]).cuda()
        self.boundaries = boundaries # torch.tensor([39., 67, 98, 132, 167, 201, 235, 272, 311, 351, 395, 442, 491, 542, 599, 659, 721, 793, 875, 965, 1065, 1180, 1323, 1496, 1703, 1951, 2269, 2659, 3441, 10000]).cuda()
        self.embed = dist_embed # torch.nn.Embedding(len(self.boundaries) + 1, dist_dim)
        self.G = nn.Linear(dist_dim, out_feats, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.tanh = nn.Tanh() # nn.Tanh(); n.LeakyReLU(0.2)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(
                    in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh') # tanh; relu
        
        if self.transform:
            nn.init.xavier_normal_(self.fc.weight)
        # nn.init.xavier_normal_(self.attn_out.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc)
        # nn.init.xavier_normal_(self.attn_fc.weight)
        nn.init.xavier_normal_(self.agg_fc.weight)

        nn.init.xavier_normal_(self.G.weight)
        # nn.init.xavier_normal_(self.combine.weight, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))
            
    def combine_dist(self, edges):
        dist = (edges.dst['pos'] - edges.src['pos']).norm(p=2, dim=1)
        dist_embed = self.G(self.embed(torch.bucketize(dist, self.boundaries)))
        h_dist = self.combine(dist_embed * edges.src['h'])
        return {'dh': h_dist}
    
    def combine_double_dist(self, edges):
        dist1 = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
        # dist_ = (edges.src['loc'].unsqueeze(1).expand([-1, num, 2]) - edges.data['inter_pos']).norm(p=2, dim=1)
        dist_ = (edges.src['loc'].unsqueeze(1).repeat(1,self.num,1) - edges.data['inter_pos']).norm(p=2, dim=-1)
        dist_embed1 = self.G(self.embed(torch.bucketize(dist1, self.boundaries)))
        dist_embed_ = self.G(self.embed(torch.bucketize(dist_, self.boundaries)).view(-1, self.dist_dim))
        
        dist_embed_dot_h = (dist_embed_.view(-1, self.num, self.out_feats) * edges.data['inter_h']).mean(dim=1)
        # dist_embed_dot_h = (dist_embed_.view(-1, self.num, self.out_feats) * edges.data['inter_h']).max(dim=1)[0]
        # dist_embed_dot_h = dist_embed_.view(-1, self.num, self.out_feats) * edges.data['inter_h']
        # src_with_dist = (dist_embed1 * edges.src['h']).unsqueeze(1).repeat(1,self.num,1)
        # weights = torch.sum(dist_embed_dot_h * src_with_dist, dim=-1).view(-1,self.num,1) / np.sqrt(self.out_feats)
        # src_with_dist = torch.cat([edges.src['h'], dist_embed1], dim=-1).unsqueeze(1).repeat(1,self.num,1)
        # inter_with_dist = torch.cat([edges.data['inter_h'], dist_embed_.view(-1, self.num, self.out_feats)], dim=-1)
        # weights = torch.sum(inter_with_dist * src_with_dist, dim=-1).view(-1,self.num,1) / np.sqrt(2 * self.out_feats)
        # dist_embed_dot_h = (F.softmax(weights,dim=1) * dist_embed_dot_h).sum(dim=1)

        # dist_embed_dot_h = dist_embed_.view(-1, self.num, self.out_feats) * edges.data['inter_h']
        # src_with_dist = (dist_embed1 * edges.src['h']).unsqueeze(1).repeat(1,self.num,1)
        # weights = self.attn_fc(torch.cat([src_with_dist, dist_embed_dot_h], dim=-1).view(-1, 2 * self.out_feats))
        # weights = self.attn_out(self.tanh(weights)).view(-1, self.num, 1)
        # dist_embed_dot_h = (F.softmax(weights,dim=1) * dist_embed_dot_h).sum(dim=1)

        h_dist = edges.dst['d0'] * edges.src['d2'] * self.agg_fc(torch.cat([dist_embed1 * edges.src['h'], dist_embed_dot_h], dim=1)) # v3
        # h_dist = edges.dst['d0'] * edges.src['d2'] * (edges.src['h'] + self.attn_fc(torch.cat([dist_embed1, dist_embed_, edges.data['inter_h']], dim=1))) # v5, gated/pooling/new weight
        return {'h': h_dist}
    
    def building_inds_random(self, graph, num=1):
        cuda = graph.device
        graph = graph.to('cpu')
        src, dst = graph.edges()
        sg = dgl.sampling.sample_neighbors(graph, dst, num, replace=True)
        src_new = sg.edges()[0].view(-1, num)
        return src_new.to(cuda)

    def building_inds(self, graph):
        src, dst = graph.edges()
        max_ids = scatter(src, dst, reduce="max")
        return max_ids[dst]

    def edge_attention(self, edges):
        # h_c = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        # h_c = torch.cat([edges.data['dh'], edges.dst['h']], dim=1)
        h_c = torch.cat([edges.data['d1'],edges.data['d2'],edges.data['d3']], dim=1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {'e': h_s}
    
    def message_func(self, edges):
        # return {'h': edges.data['dh'], 'e': edges.data['e']}
        return {'h': edges.data['h'], 't': edges.data['_TYPE']}

    def reduce_func(self, nodes):
        times = nodes.mailbox['t'] # .squeeze(2)
        time_h = torch_scatter.scatter(nodes.mailbox['h'], times, dim=1, dim_size=3)
        decay = torch.FloatTensor([1,1/np.e,1/np.e]).cuda()
        time_s = time_h.sum(dim=-1)
        attn_h = torch.bmm(time_h.transpose(1,0), self.attn_fc).transpose(1,0)
        # attn_h = self.attn_fc(time_h.view(-1, self.out_feats))
        attn_h = self.tanh(attn_h)
        weight = self.attn_out(attn_h).view(-1, 3)
        # weight = weight * decay
        weight = torch.where(time_s==0, torch.ones_like(time_s).cuda()*1e-9, weight)
        weight = F.softmax(weight, dim=1).view(-1, 3, 1)
        # weight = weight * decay
        # weight = decay.view(-1, 1)
        # ft = torch.sum(3 * weight * time_h, dim=1)
        ft = torch.sum(time_h, dim=1)
        return {'ft': ft}
    
    def forward(self, graph, feat):
        graph = graph.local_var()
        # feat = self.feat_drop(feat)
        if self.transform:
            feat = self.fc(feat)

        graph.ndata['h'] = feat
        locations = graph.ndata['loc']

        inter_ids = self.building_inds_random(graph, num=self.num)
        graph.edata['inter_pos'] = locations[inter_ids]
        graph.edata['inter_h'] = feat[inter_ids]
        
        graph.ndata['d0'] = torch.pow(graph.in_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.ndata['d2'] = torch.pow(graph.out_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.apply_edges(self.combine_double_dist)

        graph.update_all(fn.copy_e('h', 'm'),
                            fn.sum('m', 'ft'))
        # graph.update_all(self.message_func, self.reduce_func)
        rst = graph.dstdata['ft']

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

# self-attention
class DistAttn(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dist_dim,
                 boundaries,
                 dist_embed,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=None):
        super(DistAttn, self).__init__()
        self.out_dim = np.sqrt(out_feats)
        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        self.attn_wq = nn.Linear(out_feats, out_feats, bias=False)
        self.attn_wk = nn.Linear(out_feats, out_feats, bias=False)
        self.attn_wq2 = nn.Linear(out_feats * 2, out_feats, bias=False)
        self.attn_wk2 = nn.Linear(out_feats * 2, out_feats, bias=False)
        self.boundaries = boundaries
        self.embed = dist_embed
        self.G = nn.Linear(dist_dim, out_feats, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.relu = nn.ReLU() # nn.Tanh(); n.LeakyReLU(0.2)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.attn_wq.weight)
        nn.init.xavier_normal_(self.attn_wk.weight)
        nn.init.xavier_normal_(self.attn_wq2.weight)
        nn.init.xavier_normal_(self.attn_wk2.weight)
        nn.init.xavier_normal_(self.G.weight)

    def combine_double_dist(self, edges):
        dist1 = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
        dist_ = (edges.src['loc'] - edges.data['inter_pos']).norm(p=2, dim=1)
        dist_embed1 = self.G(self.embed(torch.bucketize(dist1, self.boundaries)))
        dist_embed_ = self.G(self.embed(torch.bucketize(dist_, self.boundaries)))
        q = self.attn_wq(edges.dst['feat'])
        k = self.attn_wk(edges.src['feat'])
        e1 = torch.sum(q * k, dim=1).view(-1, 1) / self.out_dim
        q = self.attn_wq2(torch.cat([edges.src['feat'], dist_embed1], dim=1))
        k = self.attn_wk2(torch.cat([edges.data['inter_h'], dist_embed_], dim=1))
        e2 = 1.0 * torch.sum(q * k, dim=1).view(-1, 1) / self.out_dim
        return {'e': e1}
    
    def building_inds_random(self, graph, num=1):
        cuda = graph.device
        graph = graph.to('cpu')
        src, dst = graph.edges()
        sg = dgl.sampling.sample_neighbors(graph, dst, num)
        src_new = sg.edges()[0]
        return src_new.to(cuda)

    def building_inds(self, graph):
        src, dst = graph.edges()
        max_ids = scatter(src, dst, reduce="max")
        return max_ids[dst]

    def edge_attention(self, edges):
        # h_c = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        # h_c = torch.cat([edges.data['dh'], edges.dst['h']], dim=1)
        h_c = torch.cat([edges.data['d1'],edges.data['d2'],edges.data['d3']], dim=1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {'e': h_s}
    
    def forward(self, graph, feat):
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        feat_h = self.fc(feat)

        graph.ndata['feat'] = feat
        graph.ndata['h'] = feat_h
        locations = graph.ndata['loc']

        inter_ids = self.building_inds_random(graph)
        graph.edata['inter_pos'] = locations[inter_ids]
        graph.edata['inter_h'] = feat[inter_ids]
        graph.apply_edges(self.combine_double_dist)

        e = graph.edata.pop('e')
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                            fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

class MultiHeadDistGATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dist_dim,
                 boundaries,
                 dist_embed,
                 num_heads=3,
                 feat_drop=0.,
                 attn_drop=0.,
                 merge='mean',
                 residual=False,
                 transform=False,
                 activation=None):
        super(MultiHeadDistGATLayer, self).__init__()
        self.transform = transform
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.merge = merge
        if self.transform:
            self.fc = nn.Linear(in_feats, out_feats*num_heads, bias=False)
            attn_in_feats = 2 * out_feats
        else:
            attn_in_feats = 2 * in_feats
        self.attn_fc = nn.Parameter(torch.FloatTensor(size=(num_heads, 3*out_feats, out_feats)))
        self.attn_out = nn.Parameter(torch.FloatTensor(size=(num_heads, out_feats, 1)))

        self.attn_fc1 = nn.Parameter(torch.FloatTensor(size=(num_heads, 2*out_feats, out_feats)))
        self.attn_fc2 = nn.Parameter(torch.FloatTensor(size=(num_heads, 2*out_feats, out_feats)))
        self.attn_fc3 = nn.Parameter(torch.FloatTensor(size=(num_heads, 2*out_feats, out_feats)))
        # dist_dim = out_feats
        
        # self.boundaries = torch.tensor([98., 201, 311, 442, 599, 793, 1065, 1496, 2269, 10000]).cuda()
        self.boundaries = boundaries # torch.tensor([39., 67, 98, 132, 167, 201, 235, 272, 311, 351, 395, 442, 491, 542, 599, 659, 721, 793, 875, 965, 1065, 1180, 1323, 1496, 1703, 1951, 2269, 2659, 3441, 10000]).cuda()
        self.embed = dist_embed # torch.nn.Embedding(len(self.boundaries) + 1, dist_dim)
        self.G = nn.Parameter(torch.FloatTensor(size=(num_heads, dist_dim, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.tanh = nn.Tanh() # nn.Tanh(); n.LeakyReLU(0.2)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(
                    in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh') # tanh; relu
        
        if self.transform:
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_out, gain=gain)
        nn.init.xavier_normal_(self.attn_fc1, gain=gain)
        nn.init.xavier_normal_(self.attn_fc2, gain=gain)
        nn.init.xavier_normal_(self.attn_fc3, gain=gain)
        nn.init.xavier_normal_(self.attn_fc, gain=gain)

        nn.init.xavier_normal_(self.G, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))
            
    def combine_dist(self, edges):
        dist = (edges.dst['pos'] - edges.src['pos']).norm(p=2, dim=1)
        dist_embed = self.G(self.embed(torch.bucketize(dist, self.boundaries)))
        h_dist = self.combine(dist_embed * edges.src['h'])
        return {'dh': h_dist}
    
    def combine_double_dist(self, edges):
        src_pos = edges.src['pos'].unsqueeze(1).repeat(1,self.num_heads,1)
        dst_pos = edges.dst['pos'].unsqueeze(1).repeat(1,self.num_heads,1)
        dist1 = (dst_pos - src_pos).norm(p=2, dim=-1)
        dist2 = (dst_pos - edges.data['inter_pos']).norm(p=2, dim=-1)
        dist_ = (src_pos - edges.data['inter_pos']).norm(p=2, dim=-1)
        dist_embed1 = self.embed(torch.bucketize(dist1, self.boundaries)).transpose(1,0)
        dist_embed2 = self.embed(torch.bucketize(dist2, self.boundaries)).transpose(1,0)
        dist_embed_ = self.embed(torch.bucketize(dist_, self.boundaries)).transpose(1,0)
        dist_embed1 = torch.bmm(dist_embed1, self.G)
        dist_embed2 = torch.bmm(dist_embed2, self.G)
        dist_embed_ = torch.bmm(dist_embed_, self.G)

        # dist_embed1 = self.embed(torch.bucketize(dist1, self.boundaries))
        # dist_embed2 = self.embed(torch.bucketize(dist2, self.boundaries))
        # dist_embed_ = self.embed(torch.bucketize(dist_, self.boundaries))
        # h_dist = torch.cat([edges.src['h'] * dist_embed1, edges.data['inter_h'] * dist_embed2, dist_embed_], dim=1)
        # h_dist = torch.cat([edges.src['h'], dist_embed1], dim=1)
        # h_dist = self.combine(h_dist)
        # h_dist = edges.src['h'] + h_dist
        # h_dist = self.combine(dist_embed * edges.src['h'])
        d1 = torch.cat([edges.src['h'].transpose(1,0), dist_embed1], dim=-1)
        d2 = torch.cat([edges.data['inter_h'].transpose(1,0), dist_embed2], dim=-1)
        d3 = torch.cat([edges.dst['h'].transpose(1,0), dist_embed_], dim=-1)
        d1 = torch.bmm(d1, self.attn_fc1)
        d2 = torch.bmm(d2, self.attn_fc2)
        d3 = torch.bmm(d3, self.attn_fc3)
        return {'d1':d1.transpose(1,0), 'd2':d2.transpose(1,0), 'd3':d3.transpose(1,0)}
    
    def building_inds_random(self, graph, num=1):
        cuda = graph.device
        graph = graph.to('cpu')
        src, dst = graph.edges()
        sg = dgl.sampling.sample_neighbors(graph, dst, num, replace=True)
        src_new = sg.edges()[0].reshape(-1, num)
        return src_new.to(cuda)

    def building_inds(self, graph):
        src, dst = graph.edges()
        max_ids = scatter(src, dst, reduce="max")
        return max_ids[dst]

    def edge_attention(self, edges):
        # h_c = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        # h_c = torch.cat([edges.data['dh'], edges.dst['h']], dim=1)
        h_c = torch.cat([edges.data['d1'],edges.data['d2'],edges.data['d3']], dim=-1)
        h_c = torch.bmm(h_c.transpose(1,0), self.attn_fc)
        h_c = self.tanh(h_c)
        h_s = torch.bmm(h_c, self.attn_out).transpose(0,1) # num_heads, bs, 1
        return {'e': h_s}
    
    def message_func(self, edges):
        # return {'h': edges.data['dh'], 'e': edges.data['e']}
        return {'h': edges.src['h'], 'e': edges.data['e']}

    def my_message_func(self, edges):
        # return {'h': edges.data['dh'], 'e': edges.data['e']}
        e = edges.data['a'] * edges.data['dh']
        # e = edges.data['a'] * edges.src['h']
        return {'m': e}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.attn_drop(alpha)
        z = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'z': z}
    
    def forward(self, graph, feat, locations, relations):
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        if self.transform:
            feat = self.fc(feat).view(-1, self.num_heads, self.out_feats)

        graph.ndata['h'] = feat
        locations = locations.squeeze()
        graph.ndata['pos'] = locations

        inter_ids = self.building_inds_random(graph, self.num_heads)
        graph.edata['inter_pos'] = locations[inter_ids]
        inter_ids = inter_ids.unsqueeze(-1).repeat((1, 1, self.out_feats))
        graph.edata['inter_h'] = torch.gather(feat, 0, inter_ids) # bs, num_heads, dim
        graph.apply_edges(self.combine_double_dist)
        graph.apply_edges(self.edge_attention)
        # graph.update_all(self.message_func, self.reduce_func)
        # rst = graph.ndata.pop('h')

        e = graph.edata.pop('e')
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        # graph.apply_edges(self.my_message_func)
        graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                            fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(feat).view(feat.shape[0], -1, self._out_feats)
            rst = rst + resval
        if self.merge == 'mean':
            rst = torch.mean(rst, dim=1)
        if self.merge == 'cat':
            rst = rst.view(-1, self.num_heads * self.out_feats)
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

class GATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=False,
                 transform=False,
                 activation=None):
        super(GATLayer, self).__init__()
        self.transform = transform
        if self.transform:
            self.fc = nn.Linear(in_feats, out_feats, bias=False)
            attn_in_feats = 2 * out_feats
        else:
            attn_in_feats = 2 * in_feats
        self.attn_fc = nn.Linear(attn_in_feats, out_feats, bias=True)
        self.attn_out = nn.Linear(out_feats, 1, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.tanh = nn.Tanh()
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(
                    in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh')
        
        if self.transform:
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_out.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))
            
    def edge_attention(self, edges):
        h_c = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {'e': h_s}
    
    def message_func(self, edges):
        return {'h': edges.src['h'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # print(nodes.mailbox['e'].shape)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.attn_drop(alpha)
        z = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'z': z}
    
    def forward(self, graph, feat):
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        if self.transform:
            feat = self.fc(feat)
        
        graph.ndata['h'] = feat
        graph.apply_edges(self.edge_attention)
        graph.update_all(self.message_func, self.reduce_func)
        rst = graph.ndata.pop('h')

        # e = graph.edata.pop('e')
        # # compute softmax
        # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # # message passing
        # graph.update_all(fn.u_mul_e('h', 'a', 'm'),
        #                     fn.sum('m', 'ft'))
        # rst = graph.dstdata['ft']

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(feat).view(feat.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst



class HeteroGraphConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation
        Invoke the forward function with each module and aggregate their results.
        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.
        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype[-2:]](
                    rel_graph,
                    src_inputs[stype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype[-2:]](
                    rel_graph,
                    inputs[stype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)

def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.
    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.
    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        raise DGLError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)
