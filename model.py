import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import SpatialEvoConv, MLPClassifier, TimeDiscriminator


class SEENet(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_neighbor, dropout=0, reg_param=0, reg_grid=0, boundaries=None):
        super(SEENet, self).__init__()
        dist_dim = h_dim
        self.reg_param = reg_param
        self.reg_grid = reg_grid
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.boundaries = boundaries
        self.dist_embed = torch.nn.Embedding(len(self.boundaries) + 1, dist_dim)

        self.discriminator = TimeDiscriminator(h_dim)
        self.embedding = torch.nn.Embedding(in_dim, h_dim) 
        self.gnn = SpatialEvoConv(h_dim, h_dim, dist_dim, num_neighbor, self.boundaries, self.dist_embed, dropout, activation=F.relu, hop1_fc=False, elem_gate=False, merge='sum')
        self.mlp = nn.ModuleList()
        for _ in range(4):
            self.mlp.append(MLPClassifier(h_dim * 2, h_dim, activation=F.relu))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
    
    def calc_evo_score(self, embedding, embedding_, pairs, idx):
        pair_embed = embedding[pairs[:,0]] * embedding[pairs[:,1]]
        pair_embed_ = embedding_[pairs[:,0]] * embedding_[pairs[:,1]]
        score = self.mlp[idx](torch.cat([pair_embed, pair_embed_], dim=-1))
        return score

    def forward(self, g, h):
        h = self.embedding(h.squeeze())
        h = self.gnn.forward(g, h)
        return h

    def get_loss(self, g, embed, evo_pairs, evo_labels, grid_sizes, pos_samples, neg_samples, grid_labels):
        predict_loss = 0
        for idx in range(4):
            evo_score = self.calc_evo_score(embed[idx], embed[(idx+1)%4], evo_pairs[idx], idx)
            logits = self.discriminator(embed[idx], embed[(idx+1)%4], grid_sizes, pos_samples, neg_samples)
            predict_loss += self.reg_grid * F.binary_cross_entropy_with_logits(logits, grid_labels)
            predict_loss += self.reg_param * F.binary_cross_entropy_with_logits(evo_score, evo_labels[idx])
        w_dist = self.dist_embed.weight
        predict_loss += 0.000001 * (torch.sum((w_dist[1:, :] - w_dist[:-1, :])**2))
        return predict_loss


class SEENetPred(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_neighbor, dropout=0, boundaries=None):
        super(SEENetPred, self).__init__()
        dist_dim = h_dim
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.boundaries = boundaries
        self.dist_embed = torch.nn.Embedding(len(self.boundaries) + 1, dist_dim)

        self.embedding = torch.nn.Embedding(in_dim, h_dim) 
        self.gnn = SpatialEvoConv(h_dim, h_dim, dist_dim, num_neighbor, self.boundaries, self.dist_embed, dropout, activation=F.relu, hop1_fc=False, elem_gate=False, merge='sum')
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score
    
    def filter_o(self, triplets_to_filter, target_s, target_r, target_o, train_ids):
        target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
        filtered_o = []
        if (target_s, target_r, target_o) in triplets_to_filter:
            triplets_to_filter.remove((target_s, target_r, target_o))
        for o in train_ids:
            if (target_s, target_r, o) not in triplets_to_filter:
                filtered_o.append(o)
        return torch.LongTensor(filtered_o).cuda()

    def rank_score_filtered(self, embedding, test_triplets, train_triplets, valid_triplets):
        with torch.no_grad():
            s = test_triplets[:, 0]
            r = test_triplets[:, 1]
            o = test_triplets[:, 2]
            test_size = test_triplets.shape[0]
            triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()
            train_ids = torch.unique(train_triplets[:,[0,2]]).tolist()
            triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
            num_entities = embedding.shape[0]
            ranks = []

            for idx in range(test_size):
                target_s = s[idx]
                target_r = r[idx]
                target_o = o[idx]

                filtered_o = self.filter_o(triplets_to_filter, target_s, target_r, target_o, train_ids)
                if len((filtered_o == target_o).nonzero()) == 0:
                    continue
                target_o_idx = int((filtered_o == target_o).nonzero())
                emb_s = embedding[target_s]
                emb_r = self.w_relation[target_r]
                emb_o = embedding[filtered_o]
                emb_triplet = emb_s * emb_r * emb_o
                scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
                _, indices = torch.sort(scores, descending=True)
                rank = int((indices == target_o_idx).nonzero())
                ranks.append(rank)

        return np.array(ranks)

    def forward(self, g, h):
        h = self.embedding(h.squeeze())
        h = self.gnn.forward(g, h)
        return h

    def get_loss(self, g, embed, triplets, labels):
        predict_loss = 0
        for idx in range(4):
            score = self.calc_score(embed[idx], triplets[idx])
            predict_loss += F.binary_cross_entropy_with_logits(score, labels[idx])
        return predict_loss