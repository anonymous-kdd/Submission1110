import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataset import LocationRelDataset
from model import SEENetPred
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
    valid_data = data.valid
    test_data = data.test
    coords = data.coords
    relation_dict = data.relation_dict
    num_rels = data.num_rels
    grid_dict = data.grid

    boundaries = utils.adaptive_bin_distance(args.bin_num, coords, train_data)

    log = str(args)+'\n'
    # create model
    model = SEENetPred(num_nodes, args.n_hidden, num_rels,
                       boundaries=boundaries,
                       num_neighbor=args.n_neighbor,
                       dropout=args.dropout)

    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path)
        model_now_dict = model.state_dict()
        for k, v in checkpoint['state_dict'].items():
            if 'discriminator' in k or 'mlp' in k:
                continue
            model_now_dict[k] = v
        model.load_state_dict(model_now_dict)
        print("Loading pre-trained model for epoch: {}".format(checkpoint['epoch']))

    # validation and testing triplets
    valid_data = torch.from_numpy(valid_data)
    test_data = torch.from_numpy(test_data)

    hop2_dict = utils.build_hop2_dict(train_data, relation_dict, args.dataset, path_len=2)
    test_graph = utils.generate_sampled_hetero_graphs_and_labels(train_data, num_nodes, relation_dict, hop2_dict, coords=coords, test=True)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    if use_cuda:
        model.cuda()
        test_graph = test_graph.to(args.gpu)
        node_id = node_id.cuda()
        valid_data_list = valid_data.cuda()
        test_data_list = test_data.cuda()
        valid_data_list = utils.select_time_data(valid_data_list, relation_dict)
        test_data_list = utils.select_time_data(test_data_list, relation_dict)
        train_data_list = utils.select_time_data(torch.from_numpy(train_data).cuda(), relation_dict)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)
    model_state_file = 'runs/%s.pth' % (args.run_name)

    # training loop
    print("start training...")
    epoch = 0
    best_mrr = 0
    best_metrics = 0
    while True:
        model.train()
        epoch += 1
        g, data, labels = utils.generate_sampled_hetero_graphs_and_labels(train_data, num_nodes, relation_dict,
                                                                         hop2_dict=hop2_dict,
                                                                         coords=coords,
                                                                         negative_rate=args.negative_sample,
                                                                         split_size=0.8)

        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        if use_cuda:
            coord = coords[node_id].cuda()
            data, labels = data.cuda(), labels.long().cuda().view(-1, 1)
            g = g.to(args.gpu)

        embed = model(g, node_id)
        data_labels = torch.cat([data, labels], dim=1)
        data_labels = utils.select_time_data(data_labels, relation_dict)
        data, labels = [data_labels[i][:, :3] for i in range(4)], [data_labels[i][:, 3].float() for i in range(4)]
        loss = model.get_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        train_loss = loss.item()
        print("Epoch {:04d} | Loss {:.4f} | Best Valid MRR@10 {:.4f}".format(epoch, train_loss, best_mrr))
        log += "Epoch {:04d} | Loss {:.4f}  | Best Valid MRR@10 {:.4f}\n".format(epoch, train_loss, best_mrr)
        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            model.eval()
            # print("start eval")
            valid_metrics_list = []
            with torch.no_grad():
                embed = model(test_graph, node_id)
                for ii, (trn_data, tst_data, val_data) in enumerate(zip(train_data_list, test_data_list, valid_data_list)):
                    valid_ranks = model.rank_score_filtered(embed[ii], val_data, trn_data, tst_data)
                    valid_metrics = utils.mrr_hit_metric(valid_ranks)
                    valid_metrics_list.append(valid_metrics)
                    # print(valid_metrics)
                    log += str(valid_metrics) + '\n'
            
            valid_metrics = utils.overall_mrr_hit_metric(args.data_dir, args.dataset, valid_metrics_list)
            # save best model
            if best_mrr < valid_metrics['MRR@10']:
                best_mrr = valid_metrics['MRR@10']
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            
            if epoch >= args.n_epochs:
                break

    print("training done")
    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    model.eval()
    with torch.no_grad():
        model.load_state_dict(checkpoint['state_dict'])
        print("Using best epoch: {}".format(checkpoint['epoch']))
        embed = model(test_graph, node_id)
        test_metrics_list = []
        for ii, (trn_data, tst_data, val_data) in enumerate(zip(train_data_list, test_data_list, valid_data_list)):
            test_ranks = model.rank_score_filtered(embed[ii], tst_data, trn_data, val_data)
            test_metrics = utils.mrr_hit_metric(test_ranks)
            test_metrics_list.append(test_metrics)
            # print(test_metrics)
            log += str(test_metrics) + '\n'
        test_metrics = utils.overall_mrr_hit_metric(args.data_dir, args.dataset, test_metrics_list)

    print_result = ''
    for k, v in test_metrics.items():
        print_result += 'Test {:s} {:.4f} | '.format(k, v)
    print(print_result)

    log += str(print_result) + '##' + args.run_name
    f = open(f'logs/{args.run_name}.txt', 'w')
    f.write(log)
    f.close()
    f = open(f'output/{args.run_name}.txt', 'w')
    f.write(log)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trial')
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-neighbor", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=3000)
    parser.add_argument("-d", "--dataset", type=str, default='tokyo')
    parser.add_argument("--pretrain-path", type=str, default='ssl_model')
    parser.add_argument("--grid-len", type=int, default=300)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--negative-sample", type=int, default=5)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--bin_num", type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)

    run_name = 'trial_model'
    run_name = args.name + "_" + run_name
    args.run_name = f'{args.dataset}_{run_name}'
    main(args)