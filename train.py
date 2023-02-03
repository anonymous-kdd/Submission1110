import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataset import KnowledgeGraphDataset
from dgl.nn.pytorch import RelGraphConv, GraphConv, GATConv
from sklearn.metrics import f1_score
from mymodel_new import GATLayer, DistGATLayer, MultiHeadDistGATLayer, DistRelConv, Discriminator, TimeDiscriminator
import os
import utils_new as utils
import wandb
torch.set_num_threads(1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, activation, bias=True):
        super(MLPClassifier, self).__init__()
        self.bias = bias
        self.activation = activation
        self.fc_h = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc_o = nn.Linear(hidden_dim, 1, bias=bias)
        # self.fc_o = nn.Linear(in_dim, 1, bias=bias)
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
        # return self.fc_o(input_feat)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_neighbor, dist_dim=128, mlp=False, transform=True, update=True,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, reg_grid=0, gnn_model='dgn', boundaries=None):
        super(LinkPredict, self).__init__()
        dist_dim = h_dim
        self.reg_param = reg_param
        self.reg_grid = reg_grid
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.boundaries = boundaries
        self.dist_embed = torch.nn.Embedding(len(self.boundaries) + 1, dist_dim)

        # self.discriminator = Discriminator(h_dim)
        self.discriminator = TimeDiscriminator(h_dim)
        self.embedding = torch.nn.Embedding(in_dim, h_dim) 
        self.gnn = DistRelConv(h_dim, h_dim, dist_dim, num_neighbor, self.boundaries, self.dist_embed, dropout, activation=F.relu, hop1_fc=False, elem_gate=False, merge='sum')
        self.mlp = nn.ModuleList()
        for _ in range(4):
            self.mlp.append(MLPClassifier(h_dim * 2, h_dim, activation=F.relu))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score
    
    def calc_evo_score(self, embedding, embedding_, pairs, idx):
        pair_embed = embedding[pairs[:,0]] * embedding[pairs[:,1]]
        pair_embed_ = embedding_[pairs[:,0]] * embedding_[pairs[:,1]]
        score = self.mlp[idx](torch.cat([pair_embed, pair_embed_], dim=-1))
        return score
    
    def filter_o(self, triplets_to_filter, target_s, target_r, target_o, train_ids):
        target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
        filtered_o = []
        # Do not filter out the test triplet, since we want to predict on it
        if (target_s, target_r, target_o) in triplets_to_filter:
            triplets_to_filter.remove((target_s, target_r, target_o))
        # Do not consider an object if it is part of a triplet to filter
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
                #if idx % 100 == 0:
                #    print("test triplet {} / {}".format(idx, test_size))
                target_s = s[idx]
                target_r = r[idx]
                target_o = o[idx]

                filtered_o = self.filter_o(triplets_to_filter, target_s, target_r, target_o, train_ids)
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

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))
 
    def get_loss(self, g, embed, evo_pairs, evo_labels, grid_sizes, pos_samples, neg_samples, grid_labels):
        predict_loss = 0
        for idx in range(4):
            evo_score = self.calc_evo_score(embed[idx], embed[(idx+1)%4], evo_pairs[idx], idx)
            # print(score.dtype, labels[idx].dtype)
            # logits = self.discriminator(embed[idx], grid_sizes, pos_samples, neg_samples)
            logits = self.discriminator(embed[idx], embed[(idx+1)%4], grid_sizes, pos_samples, neg_samples)
            predict_loss += self.reg_grid * F.binary_cross_entropy_with_logits(logits, grid_labels)
            predict_loss += self.reg_param * F.binary_cross_entropy_with_logits(evo_score, evo_labels[idx])

        w_dist = self.dist_embed.weight
        predict_loss += 0.000001 * (torch.sum((w_dist[1:, :] - w_dist[:-1, :])**2))
        return predict_loss # + 0.00001 * reg_loss


def lr_scheduler(optimizer, decrease_rate=0.9):
    """Decay learning rate by a factor."""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decrease_rate
    return optimizer

def adaptive_bin_distance(bin_num, coords, train):
    src_coord = coords[train[:, 0]] 
    dst_coord = coords[train[:, 2]] 
    delat = src_coord - dst_coord
    all_dist = np.sqrt((delat * delat).sum(1))

    # all_dist = [min(x, 10000) for x in all_dist]
    all_dist = sorted(all_dist)
    cnt = len(all_dist)
    res = []
    for i in range(0, cnt, int(cnt/bin_num)):
        res += [all_dist[i]]
    return torch.tensor(res[1:]).cuda()

def adaptive_time_bin_distance(bin_num, coords, train, relation_dict):
    result = []
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in relation_dict.items() if key in k]
        data_selected = [train[train[:, 1]==i] for i in ids]
        data_selected = np.concatenate(data_selected)
        src_coord = coords[data_selected[:, 0]] 
        dst_coord = coords[data_selected[:, 2]]
        delat = src_coord - dst_coord
        all_dist = np.sqrt((delat * delat).sum(1))

        # all_dist = [min(x, 10000) for x in all_dist]
        all_dist = sorted(all_dist)
        cnt = len(all_dist)
        res = []
        for i in range(0, cnt, int(cnt/bin_num)):
            res += [all_dist[i]]
        result.append(torch.tensor(res[1:]).cuda())
    return result

def select_time_data(data, relation_dict, key='morning'):
    result = []
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in relation_dict.items() if key in k]
        data_selected = [data[data[:, 1]==i] for i in ids]
        result.append(torch.cat(data_selected))
    return result

def main(args):
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    data = KnowledgeGraphDataset(name=args.dataset,grid_len=args.grid_len,raw_dir=args.data_dir,force_reload=True)
    num_nodes = data.num_nodes
    train_data = data.train
    coords = data.coords
    relation_dict = data.relation_dict
    num_rels = data.num_rels
    grid_dict = data.grid

    boundaries = adaptive_bin_distance(args.bin_num, coords, train_data)

    log = str(args)+'\n'
    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        dist_dim=args.dist_dim,
                        boundaries=boundaries,
                        num_neighbor=args.n_neighbor,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.evo_reg,
                        reg_grid=args.grid_reg,
                        gnn_model=args.gnn)

    if not args.debug:
        wandb.watch(model)

    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path)
        model_now_dict = model.state_dict()
        # print(checkpoint['state_dict'])
        for k, v in checkpoint['state_dict'].items():
            if 'discriminator' in k:
                continue
            model_now_dict[k] = v
        model.load_state_dict(model_now_dict)
        print("Loading pre-trained model for epoch: {}".format(checkpoint['epoch']))

    evo_data, evo_labels = utils.build_dynamic_labels(train_data, relation_dict, num_nodes)
    hop2_dict = utils.build_hop2_dict(train_data, relation_dict, args.dataset, path_len=2)
    # build test graph
    test_graph = utils.generate_sampled_hetero_graphs_and_labels(train_data, num_nodes, relation_dict, hop2_dict, coords=coords, test=True)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    if use_cuda:
        model.cuda()
        test_graph = test_graph.to(args.gpu)
        node_id = node_id.cuda()
        train_data_list = select_time_data(torch.from_numpy(train_data).cuda(), relation_dict)

    g = test_graph
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)

    dec_steps = 500
    lr_dec_rate = 0.5
    
    model_state_file = './output/%s.pth' % (args.run_name)
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    min_loss = 1e9
    stop_epoch = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        t0 = time.time()
        grid_sizes, pos_samples, neg_samples, grid_labels = utils.generate_batch_grids(grid_dict, args.grid_batch_size, args.grid_negative_ratio, sampling_hop=2)
        grid_sizes = torch.from_numpy(grid_sizes).long()
        pos_samples, neg_samples = torch.from_numpy(pos_samples).long(), torch.from_numpy(neg_samples).long()
        grid_labels = torch.from_numpy(grid_labels)
        
        if use_cuda:
            grid_sizes = grid_sizes.cuda()
            pos_samples = pos_samples.cuda()
            neg_samples = neg_samples.cuda()
            grid_labels = grid_labels.cuda()
            coord = coords[node_id].cuda()

        embed = model(g, node_id)
        loss = model.get_loss(g, embed, evo_data, evo_labels, grid_sizes, pos_samples, neg_samples, grid_labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()
        # if (epoch + 2) % dec_steps == 0:
        #     optimizer = lr_scheduler(optimizer, lr_dec_rate)

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        train_loss = loss.item()
        print("Epoch {:04d} | Loss {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, train_loss, forward_time[-1], backward_time[-1]))
        log += "Epoch {:04d} | Loss {:.4f} | Forward {:.4f}s | Backward {:.4f}s \n".\
              format(epoch, train_loss, forward_time[-1], backward_time[-1])
        optimizer.zero_grad()

        if train_loss < min_loss:
            min_loss = train_loss
            stop_epoch = 0
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        else:
            stop_epoch += 1
        if stop_epoch == 100:
            break

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=128,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-neighbor", type=int, default=5,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='beijing',
            help="dataset to use")
    parser.add_argument("--gnn", type=str, default='dgn')
    parser.add_argument("--pretrain-path", type=str, default='')
    parser.add_argument("--grid_len", type=int, default=500)
    parser.add_argument("--grid_negative_ratio", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="filtered",
            help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--evo_reg", type=float, default=0.1)
    parser.add_argument("--grid_reg", type=float, default=1e-3)
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--grid_batch_size", type=int, default=512,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="full",
            help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--data_dir", type=str, default="degree_811")
    parser.add_argument("--name", type=str, default="Default")
    parser.add_argument("--bin_num", type=int, default=20)
    parser.add_argument("--dist_dim", type=int, default=128)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=46)

    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)
    project_name = 'exp'
    if not args.debug:
        wandb.init(project=project_name)
    run_name = f'pretrained_evoreg{args.evo_reg}_gridbatch{args.grid_batch_size}_contrast{args.grid_reg}_gridneg{args.grid_negative_ratio}_gridlen{args.grid_len}_neighbor{args.n_neighbor}_bin{args.bin_num}_reg{args.regularization}_drop{args.dropout}_neg{args.negative_sample}_lr{args.lr}'
    if len(args.name) > 2:
        run_name = args.name + "_" + run_name
    if not args.debug:
        wandb.run.name = run_name
        wandb.config.update(args)
    args.run_name = f'{args.dataset}_{run_name}'
    main(args)