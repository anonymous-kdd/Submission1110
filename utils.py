import pickle
import numpy as np
import torch
from torch.multiprocessing import Queue
import dgl
import os
from collections import Counter
from tqdm import tqdm

def adaptive_bin_distance(bin_num, coords, train):
    src_coord = coords[train[:, 0]] 
    dst_coord = coords[train[:, 2]] 
    delat = src_coord - dst_coord
    all_dist = np.sqrt((delat * delat).sum(1))

    all_dist = sorted(all_dist)
    cnt = len(all_dist)
    res = []
    for i in range(0, cnt, int(cnt/bin_num)):
        res += [all_dist[i]]
    return torch.tensor(res[1:]).cuda()


def select_time_data(data, relation_dict, key='morning'):
    result = []
    for key in ['_morning', '_midday', '_night', '_late-night']:
        ids = [v for k, v in relation_dict.items() if key in k]
        data_selected = [data[data[:, 1]==i] for i in ids]
        result.append(torch.cat(data_selected))
    return result


# Utility function for building training and testing graphs
def generate_batch_grids(grid_dict, grid_batch_size, negative_ratio=2.0, sampling_hop=2):
    all_grid = np.arange(len(grid_dict))
    grid_batch_size = min(len(all_grid), grid_batch_size)
    inds = np.random.choice(all_grid, grid_batch_size)
    keys = list(grid_dict.keys())
    batch_keys = [keys[i] for i in inds]
    pos_samples = np.concatenate([grid_dict[k] for k in batch_keys])
    neg_samples = negative_sampling_grid_near(batch_keys, grid_dict, negative_ratio, negative_ratio)
    grid_sizes = [len(grid_dict[k]) for k in batch_keys]
    labels = np.zeros(len(pos_samples) * (negative_ratio + 1), dtype=np.float32)
    labels[: len(pos_samples)] = 1
    return np.array(grid_sizes), np.array(pos_samples), np.array(neg_samples), np.array(labels)


def negative_sampling_grid_near(keys, grid_dict, sampling_hop, negative_ratio):
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
    return np.concatenate(neg_samples)


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
    pair_list = [torch.from_numpy(np.concatenate(x)).cuda() for x in pair_list]
    label_list = [torch.from_numpy(np.concatenate(x)).view(-1, 1).cuda() for x in label_list]
    return pair_list, label_list


def build_hop2_dict(train_triplets, relation_dict, dataset, path_len=2):
    cache_file = './runs/%s_path_len_%d.pickle' % (dataset, path_len)
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

    samples = np.concatenate((pos_samples, neg_samples))
    _, ids = np.unique(samples, axis=0, return_index=True)
    ids.sort()
    samples = samples[ids]
    labels = labels[ids]
    return samples, labels


# Main evaluation function
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

    return results


def overall_mrr_hit_metric(data_dir, dataset, metrics_list):
    times = ['morning','midday','_night','late-night']
    cnt_l = [0, 0, 0, 0]
    with open('%s/%s/test.txt' % (data_dir, dataset)) as f:
        for line in f.readlines():
            _, rel, _ = line.split('\t')
            for i in range(4):
                if times[i] in rel:
                    cnt_l[i] += 1
    overall_metrics = {}
    for key in list(metrics_list[0].keys()):
        scores = [metrics[key] for metrics in metrics_list]
        score = sum([scores[i]*cnt_l[i]/sum(cnt_l) for i in range(len(scores))])
        overall_metrics[key] = score
        # score_str = str(round(score, 4))
        # overall_metrics[key] = score_str+'0'*(6-len(score_str))
    return overall_metrics