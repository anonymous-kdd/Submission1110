
import traceback
from _thread import start_new_thread
from functools import wraps
import pickle
import numpy as np
import torch
from torch.multiprocessing import Queue
import dgl
import os
from collections import Counter
from tqdm import tqdm


def generate_batch_grids(grid_dict, grid_batch_size, negative_ratio=2.0, sampling_hop=2):
    all_grid = np.arange(len(grid_dict))
    grid_batch_size = min(len(all_grid), grid_batch_size)
    inds = np.random.choice(all_grid, grid_batch_size)
    keys = list(grid_dict.keys())
    batch_keys = [keys[i] for i in inds]
    pos_samples = np.concatenate([grid_dict[k] for k in batch_keys])
    # neg_samples = negative_sampling_grid_random(batch_keys, grid_dict, negative_ratio)
    neg_samples = negative_sampling_grid_near(batch_keys, grid_dict, negative_ratio, negative_ratio)
    # samples = np.concatenate((pos_samples, neg_samples))
    grid_sizes = [len(grid_dict[k]) for k in batch_keys]
    labels = np.zeros(len(pos_samples) * (negative_ratio + 1), dtype=np.float32)
    labels[: len(pos_samples)] = 1
    return np.array(grid_sizes), np.array(pos_samples), np.array(neg_samples), np.array(labels)

def negative_sampling_grid_near(keys, grid_dict, sampling_hop, negative_ratio):
    # x_list = sorted([k[0] for k in grid_dict.keys()])
    # y_list = sorted([k[1] for k in grid_dict.keys()])
    all_nodes = set(np.concatenate([ll for ll in grid_dict.values()]))
    all_grids = list(grid_dict.keys())
    keys_expand = np.tile(np.expand_dims(keys, 1),(1,len(all_grids),1))
    manha = np.abs(keys_expand - all_grids).sum(-1)
    inds_x, inds_y = np.where((manha>sampling_hop) & (manha<sampling_hop+4))
    grids_expand = np.repeat(np.expand_dims(all_grids, 0), len(keys), axis=0)
    sampling_grids = grids_expand[inds_x, inds_y]

    neg_samples = []
    random_cnt = 0
    for i in range(len(keys)):
        neg_grids = sampling_grids[inds_x == i]
        if len(neg_grids) == 0:
            nodes_range = list(all_nodes - set(grid_dict[keys[i]]))
            random_cnt += 1
        else:
            nodes_range = np.concatenate([grid_dict[tuple(g)] for g in neg_grids])
        neg_nodes = np.random.choice(nodes_range, negative_ratio * len(grid_dict[keys[i]]))
        neg_samples.append(neg_nodes)
    print('Random case', random_cnt)
    return np.concatenate(neg_samples)

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhool expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

### Three Questions
# 1. src -> dst and dst -> src
# 2. positive labels contain all edegs?
# 3. repeated positive and negative samples?
def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    
    # perform edge neighbor sampling
    if sampler == "full":
        edges = np.arange(len(triplets))
        sample_size = len(triplets)
        split_size = 0.8
    elif sampler == "uniform":
        edges = sample_edge_uniform(adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()


    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                    size=split_size, replace=False)

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)
    # print(len(samples),len(labels),negative_rate)

    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

        # build DGL graph
        # print("# sampled nodes: {}".format(len(uniq_v)))
        # print("# sampled edges: {}".format(len(src) * 2))

    # else:
    #     src, rel, dst = triplets.transpose()
    #     uniq_v = np.arange(triplets.max()+1)
    #     relabeled_edges = np.stack((src, rel, dst)).transpose()
    #     # negative sampling
    #     samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
    #                                         negative_rate)
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def build_dynamic_labels(data, relation_dict, num_nodes):
    pair_list, label_list = [[] for _ in range(4)], [[] for _ in range(4)]
    for rel in ['competitive', 'complementary']:
        g_list = [np.zeros((num_nodes, num_nodes)) for _ in range(4)]
        for i, key in enumerate(['_morning', '_midday', '_night', '_late-night']):
            ids = [v for k, v in relation_dict.items() if key in k and rel in k]
            data_selected = [data[data[:, 1]==i] for i in ids]
            data_selected = np.concatenate(data_selected)
            g_list[i][data_selected[:,0], data_selected[:,2]] = 1
        
        for i in range(4):
            src_t, dst_t = g_list[i].nonzero()
            j = (i + 1) % 4
            labels = g_list[j][src_t, dst_t]
            pair_list[i].append(np.stack([src_t, dst_t], 1))
            label_list[i].append(labels)
            # pair_list[i].append(np.stack([dst_t, src_t], 1))
            # label_list[i].append(labels)
            # print(len(labels), labels.sum())
    pair_list = [torch.from_numpy(np.concatenate(x)).cuda() for x in pair_list]
    label_list = [torch.from_numpy(np.concatenate(x)).view(-1, 1).cuda() for x in label_list]
    return pair_list, label_list

def build_hop2_dict(train_triplets, relation_dict, dataset, path_len=2):
    cache_file = './tmp/%s_path_len_%d.pickle' % (dataset, path_len)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            hop2_dict = pickle.load(f)
        return hop2_dict

    hop2_dict = {}
    src, rel, dst = train_triplets.transpose()
    relation_dict_r = {v: int(k.split('_')[0]=='competitive') for k, v in relation_dict.items()}
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in relation_dict.items() if key in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        rel_t = rel[ids]
        dst_t = dst[ids]

        graph = {}
        for u, r, v in zip(src_t, rel_t, dst_t):
            if u not in graph:
                graph[u] = {}
            graph[u][v] = relation_dict_r[r]
        # graph = {}
        # for u, r, v in zip(dst_t, rel_t, src_t):
        #     if u not in graph:
        #         graph[u] = {}
        #     graph[u][v] = relation_dict_r[r]


        dd = {'00': {}, '01': {}, '10': {}, '11': {}}
        for u, u_N in graph.items():
            for v, r1 in u_N.items():
                for v2, r2 in graph[v].items():
                    if u != v2:
                        r12 = str(r1) + str(r2)
                        if (u,v2) not in dd[r12]:
                            dd[r12][(u,v2)] = []
                        dd[r12][(u,v2)] += [v]

        hop2_dict[key[1:]] = {}
        for k in dd.keys():
            hop2_dict[key[1:]][k] = {}
            for u, v in dd[k].items():
                if len(v) >= path_len:
                    hop2_dict[key[1:]][k][u] = v

    with open(cache_file, 'wb') as f:
        pickle.dump(hop2_dict, f)
    return hop2_dict

def generate_sampled_hetero_graph_and_labels(triplets, num_nodes, relation_dict, hop2_dict=None, coords=None, test=False, negative_rate=10, split_size=0.8):
    sample_size = len(triplets)
    src, rel, dst = triplets.transpose()
    # negative sampling
    if not test:
        samples, labels = negative_sampling(triplets, num_nodes, negative_rate) # TODO: for different graphs
        split_size = int(sample_size * split_size)
        graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size, replace=False)
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]

    edges_dict = {}
    mids_dict = {}
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in relation_dict.items() if key in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        dst_t = dst[ids]

        if hop2_dict:
            # tmp_g = {i: [] for i in range(num_nodes)}
            # for s, o in zip(src_t, dst_t):
            #     tmp_g[s].append(o)
            tmp_g = np.zeros((num_nodes, num_nodes))
            for s, o in zip(src_t, dst_t):
                tmp_g[s][o] = 1
            
            for r, reld in hop2_dict[key[1:]].items():
                src_t, dst_t, mid_t = [], [], []
                for (u, v2), vs in reld.items():
                    for v in vs:
                        # if v in tmp_g[u] and v2 in tmp_g[v]:
                        if tmp_g[u][v] and tmp_g[v][v2]:
                            src_t += [u]
                            dst_t += [v2]
                            mid_t += [v]

                rel_key = key[1:] + r
                edges_dict[('p', rel_key, 'p')] = (src_t, dst_t)
                mids_dict[rel_key] = torch.tensor(mid_t)
        else:
            edges_dict[('p', key[1:], 'p')] = (src_t, dst_t)

    hg = dgl.heterograph(edges_dict, num_nodes_dict={'p':num_nodes})
    hg.ndata['loc'] = coords # must be tensor type

    if not hop2_dict:
        hg = hg
        hg = dgl.to_homogeneous(hg, ndata=['loc'])
    else:
        hg.edata['mid'] = mids_dict

    if test:
        return hg
    else:
        return hg, samples, labels

def generate_sampled_hetero_graphs_degs_and_labels(triplets, num_nodes, relation_dict, hop2_dict=None, coords=None, test=False, negative_rate=10, split_size=0.8):
    sample_size = len(triplets)
    src, rel, dst = triplets.transpose()
    # negative sampling
    if not test:
        samples, labels = negative_sampling(triplets, num_nodes, negative_rate) # TODO: for different graphs
        split_size = int(sample_size * split_size)
        graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size, replace=False)
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]

    edges_dict = {}
    mids_dict = {}
    degs_dict = {}
    keys = ['_morning', '_midday', '_night', '_late-night']
    for i, key in enumerate(keys):
        
        # i1 = (i - 1) % 4
        # i2 = (i + 1) % 4
        # ids = [v for k, v in relation_dict.items() if key in k]
        # ids += [v for k, v in relation_dict.items() if keys[i1] in k]
        # ids += [v for k, v in relation_dict.items() if keys[i2] in k]
        # src_t = src[ids]
        # dst_t = dst[ids]
        # dist_l = (coords[src_t] - coords[dst_t]).norm(p=2, dim=1).numpy()
        # tmp_g = {nn: ([], []) for nn in range(num_nodes)}
        # for iii in range(len(src_t)):
        #     tmp_g[src_t[iii]][0].append(dist_l[iii])
        #     tmp_g[src_t[iii]][1].append(dst_t[iii])
        # ranks = np.zeros((num_nodes, num_nodes))
        # for nn in range(num_nodes):
        #     ranks[nn, tmp_g[nn][1]] = np.argsort(tmp_g[nn][0]) + 1
        # rank_t = [ranks[s][o] for s, o in zip(src_t, dst_t)]
        # edges_dict[('p', key[1:], 'p')] = (src_t, dst_t)

        ids = [v for k, v in relation_dict.items() if key in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        dst_t = dst[ids]
        edges_dict[('p', key[1:], 'p')] = (src_t, dst_t)

        if hop2_dict:
            # tmp_g = {i: [] for i in range(num_nodes)}
            # for s, o in zip(src_t, dst_t):
            #     tmp_g[s].append(o)
            tmp_g = np.zeros((num_nodes, num_nodes))
            for s, o in zip(src_t, dst_t):
                tmp_g[s][o] = 1
            
            for r, reld in hop2_dict[key[1:]].items():
                src_t, dst_t, mid_t = [], [], []
                for (u, v2), vs in reld.items():
                    for v in vs:
                        # if v in tmp_g[u] and v2 in tmp_g[v]:
                        if tmp_g[u][v] and tmp_g[v][v2]:
                            src_t += [u]
                            dst_t += [v2]
                            mid_t += [v]

                rel_key = key[1:] + r
                edges_dict[('p', rel_key, 'p')] = (src_t, dst_t)
                mids_dict[rel_key] = torch.tensor(mid_t)
                cnt_dict = Counter(mid_t)
                cnt_t = [cnt_dict[x] for x in mid_t]
                degs_dict[rel_key] = torch.tensor(cnt_t).float()


    hg = dgl.heterograph(edges_dict, num_nodes_dict={'p':num_nodes})
    hg.ndata['loc'] = coords # must be tensor type

    if not hop2_dict:
        hg = hg
        hg = dgl.to_homogeneous(hg, ndata=['loc'])
    else:
        hg.edata['mid'] = mids_dict
        hg.edata['deg'] = degs_dict

    if test:
        return hg
    else:
        return hg, samples, labels

def generate_sampled_hetero_graphs_and_labels(triplets, num_nodes, relation_dict, hop2_dict=None, coords=None, test=False, negative_rate=10, split_size=0.8):
    sample_size = len(triplets)
    src, rel, dst = triplets.transpose()
    # negative sampling
    if not test:
        samples, labels = negative_sampling(triplets, num_nodes, negative_rate) # TODO: for different graphs
        split_size = int(sample_size * split_size)
        graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size, replace=False)
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]

    edges_dict = {}
    mids_dict = {}
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in relation_dict.items() if key in k]
        ids = np.isin(rel, ids)
        src_t = src[ids]
        dst_t = dst[ids]
        edges_dict[('p', key[1:], 'p')] = (src_t, dst_t)

        if hop2_dict:
            # tmp_g = {i: [] for i in range(num_nodes)}
            # for s, o in zip(src_t, dst_t):
            #     tmp_g[s].append(o)
            tmp_g = np.zeros((num_nodes, num_nodes))
            for s, o in zip(src_t, dst_t):
                tmp_g[s][o] = 1
            
            for r, reld in hop2_dict[key[1:]].items():
                src_t, dst_t, mid_t = [], [], []
                for (u, v2), vs in reld.items():
                    for v in vs:
                        # if v in tmp_g[u] and v2 in tmp_g[v]:
                        if tmp_g[u][v] and tmp_g[v][v2]:
                            src_t += [u]
                            dst_t += [v2]
                            mid_t += [v]

                rel_key = key[1:] + r
                edges_dict[('p', rel_key, 'p')] = (src_t, dst_t)
                mids_dict[rel_key] = torch.tensor(mid_t)

    hg = dgl.heterograph(edges_dict, num_nodes_dict={'p':num_nodes})
    hg.ndata['loc'] = coords # must be tensor type

    if not hop2_dict:
        hg = hg
        hg = dgl.to_homogeneous(hg, ndata=['loc'])
    else:
        hg.edata['mid'] = mids_dict

    if test:
        return hg
    else:
        return hg, samples, labels

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    # 需要注意reverse的问题
    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    # src, dst = np.concatenate((src, dst)), np.concatenate((dst, src)) # TODO If necessary?
    # rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    # print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64'), norm.astype('int64')

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    # new method
    # import time
    # t1 = time.time()
    # pos_samples_str = ['_'.join(map(str,x)) for x in pos_samples] # list(map(str, pos_samples)) 
    # neg_samples_str = ['_'.join(map(str,x)) for x in neg_samples]  # list(map(str, neg_samples))
    # t2 = time.time()
    # ids = np.where(np.isin(neg_samples_str, pos_samples_str))[0] + len(pos_samples_str)
    # t3 = time.time()
    # ids = np.concatenate([np.arange(len(pos_samples_str)), ids])
    # t4 = time.time()
    # print(t2-t1,t3-t2,t4-t3)
    # samples = np.concatenate((pos_samples, neg_samples))
    # new method

    # old method
    samples = np.concatenate((pos_samples, neg_samples))
    _, ids = np.unique(samples, axis=0, return_index=True)
    ids.sort()
    # old method

    samples = samples[ids]
    labels = labels[ids]
    return samples, labels

#######################################################################
#
# Utility functions for evaluations (raw)
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        # perturb subject
        ranks_s = perturb_and_get_raw_rank(embedding, w, o, r, s, test_size, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_raw_rank(embedding, w, s, r, o, test_size, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider an object if it is part of a triplet to filter
    for o in range(num_entities):
        if (target_s, target_r, o) not in triplets_to_filter:
            filtered_o.append(o)
    return torch.LongTensor(filtered_o)

def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider a subject if it is part of a triplet to filter
    for s in range(num_entities):
        if (s, target_r, target_o) not in triplets_to_filter:
            filtered_s.append(s)
    return torch.LongTensor(filtered_s)

def perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb object in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_o_idx = int((filtered_o == target_o).nonzero())
        emb_s = embedding[target_s]
        emb_r = w[target_r]
        emb_o = embedding[filtered_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_o_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)

def perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb subject in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_s_idx = int((filtered_s == target_s).nonzero())
        emb_s = embedding[filtered_s]
        emb_r = w[target_r]
        emb_o = embedding[target_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_s_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[]):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        print('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)
        print('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

#######################################################################
#
# Main evaluation function
#
#######################################################################


# def mrr_hit_metric(ranks, hits=[1,3,5,10]):
#     ranks += 1 # change to 1-indexed
#     mrr = np.mean(1.0 / ranks)
#     results = {'MRR': mrr}
#     for hit in hits:
#         avg_count = np.mean((ranks <= hit))
#         results['Hit@'+str(hit)] = avg_count
#         # print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
#     return results

def mrr_hit_metric(ranks, hits=[1,3,5,10,15,20]):
    ranks += 1 # change to 1-indexed
    # mrr = np.mean(1.0 / ranks)
    results = {}
    for hit in hits:
        avg_count = np.mean((ranks <= hit))
        results['Hit@'+str(hit)] = avg_count
        if hit >= 3:
            rank_ = 1.0 / ranks
            rank_[rank_ < 1.0/hit] = 0
            results['MRR@'+str(hit)] = np.mean(rank_)

        # print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return results

def calc_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[], eval_bz=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits)
    else:
        mrr = calc_raw_mrr(embedding, w, test_triplets, hits, eval_bz)
    return mrr



def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function
