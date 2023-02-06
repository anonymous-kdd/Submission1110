import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import dgl
import dgl.function as fn
from layer_utils import HeteroGraphConv, Identity

class SpatialEvoConv(nn.Module):
    def __init__(self, in_feats, out_feats, dist_dim, num_neighbor, boundaries, dist_embed, feat_drop, activation=None, hop1_fc=False, elem_gate=False, merge='sum'):
        super(SpatialEvoConv, self).__init__()
        self.merge = merge
        self.time_list = ['morning', 'midday', 'night', 'late-night']
        self.rs_agg = HeteroGraphConv({
            '00': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate),
            '01': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate),
            '10': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate),
            '11': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc, elem_gate)},
            aggregate='sum')
        self.se_prop = SpatialEvoProp(out_feats, out_feats, dist_dim, boundaries, dist_embed, num_neighbor, 0., 0., transform=True, activation=activation)

    def forward(self, graph, feat):
        graph = graph.local_var()
        h_list = []
        for key in self.time_list:
            etype_list = [key+ids for ids in ['00', '01', '10', '11']]
            subg = graph.edge_type_subgraph(etype_list)
            h = self.rs_agg(subg, {'p':feat})
            h_list.append(h['p'])

        h_t_list = []
        for i, key in enumerate(self.time_list):
            i1 = (i - 1) % 4
            i2 = (i + 1) % 4
            etype_list = [e for e in graph.etypes if key == e]
            etype_list += [e for e in graph.etypes if self.time_list[i1] == e]
            etype_list += [e for e in graph.etypes if self.time_list[i2] == e]
            h_t = torch.stack([h_list[i1], h_list[i], h_list[i2]],dim=1)
            if self.merge == 'sum':
                h_t = torch.sum(h_t, dim=1)
            if self.merge == 'mean':
                h_t = torch.mean(h_t, dim=1)
            if self.merge == 'max':
                h_t = torch.max(h_t, dim=1)[0]

            subg = dgl.to_homogeneous(graph.edge_type_subgraph(etype_list), ndata=['loc'])
            h_t = self.se_prop(subg, h_t)
            h_t_list.append(h_t)
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
        if self.hop1_fc: 
            self.fc1 = nn.Linear(in_feats, out_feats, bias=False)
        
        self.fc_w1 = nn.Linear(2 * out_feats, out_feats, bias=False)
        self.fc_w2 = nn.Linear(2 * out_feats, out_feats, bias=False)
        if not self.elem_gate: 
            self.vec_a = nn.Linear(out_feats, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation 
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu') 
        if self.hop1_fc:
            nn.init.xavier_uniform_(self.fc1.weight)
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
        graph.apply_edges(self.cal_dist)
        feat = self.feat_drop(feat)
        h2 = self.fc2(feat)
        if self.hop1_fc:
            h1 = self.fc1(feat)
        else:
            h1 = h2

        mid_ids = graph.edata['mid']
        graph.ndata['h2'] = h2
        graph.edata['h1'] = h1[mid_ids]
        graph.ndata['d0'] = torch.pow(graph.in_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.ndata['d2'] = torch.pow(graph.out_degrees().float().clamp(min=1).view(-1, 1), -0.5)
        graph.apply_edges(self.msg_function)

        graph.update_all(message_func=fn.copy_edge('he', 'm'), reduce_func=fn.sum(msg='m', out='x'))
        feat = graph.nodes['p'].data.pop('x')
        if self.activation:
            feat = self.activation(feat)
        return feat


class SpatialEvoProp(nn.Module):
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
        super(SpatialEvoProp, self).__init__()
        self.num = num_neighbor
        self.out_feats = out_feats
        self.dist_dim = dist_dim
        self.transform = transform
        if self.transform:
            self.fc = nn.Linear(in_feats, out_feats, bias=False)
            attn_in_feats = 2 * out_feats
        else:
            attn_in_feats = 2 * in_feats
        self.agg_fc = nn.Linear(out_feats * 2, out_feats, bias=True)
        self.boundaries = boundaries
        self.embed = dist_embed
        self.G = nn.Linear(dist_dim, out_feats, bias=False)

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
            nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.agg_fc.weight)
        nn.init.xavier_normal_(self.G.weight)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))
            
    def combine_dist(self, edges):
        dist = (edges.dst['pos'] - edges.src['pos']).norm(p=2, dim=1)
        dist_embed = self.G(self.embed(torch.bucketize(dist, self.boundaries)))
        h_dist = self.combine(dist_embed * edges.src['h'])
        return {'dh': h_dist}
    
    def combine_double_dist(self, edges):
        dist1 = (edges.dst['loc'] - edges.src['loc']).norm(p=2, dim=1)
        dist_ = (edges.src['loc'].unsqueeze(1).repeat(1,self.num,1) - edges.data['inter_pos']).norm(p=2, dim=-1)
        dist_embed1 = self.G(self.embed(torch.bucketize(dist1, self.boundaries)))
        dist_embed_ = self.G(self.embed(torch.bucketize(dist_, self.boundaries)).view(-1, self.dist_dim))
        
        dist_embed_dot_h = (dist_embed_.view(-1, self.num, self.out_feats) * edges.data['inter_h']).mean(dim=1)
        h_dist = edges.dst['d0'] * edges.src['d2'] * self.agg_fc(torch.cat([dist_embed1 * edges.src['h'], dist_embed_dot_h], dim=1))
        return {'h': h_dist}
    
    def building_inds_random(self, graph, num=1):
        cuda = graph.device
        graph = graph.to('cpu')
        src, dst = graph.edges()
        sg = dgl.sampling.sample_neighbors(graph, dst, num, replace=True)
        src_new = sg.edges()[0].view(-1, num)
        return src_new.to(cuda)
    
    def message_func(self, edges):
        return {'h': edges.data['h'], 't': edges.data['_TYPE']}
    
    def forward(self, graph, feat):
        graph = graph.local_var()
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
        rst = graph.dstdata['ft']

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class TimeDiscriminator(nn.Module):
    def __init__(self, n_h):
        super(TimeDiscriminator, self).__init__()
        self.f_i = nn.Linear(n_h, n_h)
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embedding, embedding_, grid_sizes, pos_samples, neg_samples, pos_bias=None, neg_bias=None):
        embedding_ = self.f_i(embedding_)
        pos_embed = embedding_[pos_samples]
        grid_embed = dgl.ops.segment_reduce(grid_sizes, pos_embed, 'mean')
        
        embedding = self.f_i(embedding)
        pos_embed = embedding[pos_samples]
        neg_emebd = embedding[neg_samples]

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


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, activation, bias=True):
        super(MLPClassifier, self).__init__()
        self.bias = bias
        self.activation = activation
        self.fc_h = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc_o = nn.Linear(hidden_dim, 1, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_h.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        if self.bias:
            self.fc_h.bias.data.zero_()
            self.fc_o.bias.data.zero_()
    
    def forward(self, input_feat):
        h = self.activation(self.fc_h(input_feat))
        return self.fc_o(h)