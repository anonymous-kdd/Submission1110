import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataset import LocationRelDataset
from model import SEENet
import os
import utils
torch.set_num_threads(1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    data = LocationRelDataset(name=args.dataset,grid_len=args.grid_len,raw_dir=args.data_dir)
    num_nodes = data.num_nodes
    train_data = data.train
    coords = data.coords
    relation_dict = data.relation_dict
    num_rels = data.num_rels
    grid_dict = data.grid

    boundaries = utils.adaptive_bin_distance(args.bin_num, coords, train_data)

    log = str(args)+'\n'
    # create model
    model = SEENet(num_nodes, args.n_hidden, num_rels,
                   boundaries=boundaries,
                   num_neighbor=args.n_neighbor,
                   dropout=args.dropout,
                   reg_param=args.global_weight,
                   reg_grid=args.local_weight)

    evo_data, evo_labels = utils.build_dynamic_labels(train_data, relation_dict, num_nodes)
    hop2_dict = utils.build_hop2_dict(train_data, relation_dict, args.dataset, path_len=2)
    test_graph = utils.generate_sampled_hetero_graphs_and_labels(train_data, num_nodes, relation_dict, hop2_dict, coords=coords, test=True)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    if use_cuda:
        model.cuda()
        test_graph = test_graph.to(args.gpu)
        node_id = node_id.cuda()
        train_data_list = utils.select_time_data(torch.from_numpy(train_data).cuda(), relation_dict)

    g = test_graph
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)
    model_state_file = './output/%s.pth' % (args.run_name)

    # training loop
    print("start training...")
    epoch = 0
    min_loss = 1e9
    stop_epoch = 0
    while True:
        model.train()
        epoch += 1
        grid_sizes, pos_samples, neg_samples, grid_labels = utils.generate_batch_grids(grid_dict, args.global_batch_size, args.global_neg_ratio, sampling_hop=2)
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        train_loss = loss.item()
        print("Epoch {:04d} | Loss {:.4f}".format(epoch, train_loss))
        log += "Epoch {:04d} | Loss {:.4f} \n".format(epoch, train_loss)
        optimizer.zero_grad()

        if train_loss < min_loss:
            min_loss = train_loss
            stop_epoch = 0
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        else:
            stop_epoch += 1
        if stop_epoch == 100:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trial')
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-neighbor", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=2000)
    parser.add_argument("-d", "--dataset", type=str, default='tokyo')
    parser.add_argument("--pretrain-path", type=str, default='')
    parser.add_argument("--grid-len", type=int, default=300)
    parser.add_argument("--global-neg-ratio", type=int, default=3)
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--negative-sample", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--bin_num", type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)

    run_name = f'ssl_model'
    run_name = args.name + "_" + run_name
    args.run_name = f'{args.dataset}_{run_name}'
    main(args)

