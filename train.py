import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import GCN, HybridMethod
from sampler import Sampler_FastGCN, Sampler_ASGCN, Alternative_Sampler1_FastGCN,Alternative_Sampler2_FastGCN
from utils import load_data, get_batches, accuracy, f1_score
from utils import sparse_mx_to_torch_sparse_tensor

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Fast',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=10,
                        help='the train epochs between two test')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(train_ind, train_labels, batch_size, train_times):
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            
            sampled_feats, sampled_adjs, var_loss = model.sampling(
                batch_inds)
            optimizer.zero_grad()
            output = model(sampled_feats, train_features[batch_inds], sampled_adjs)
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def test(test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_feats[test_index], test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)
    preds = outputs.max(1)[1].type_as(test_labels)
    score = f1_score (preds, test_labels, outputs.shape[1])
    return loss_test.item(), acc_test.item(), time.time() - t, score


if __name__ == '__main__':
    # load data, set superpara and constant
    args = get_args()
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data(args.dataset)

    layer_sizes = [128, 128]
    input_dim = features.shape[1]
    
    train_nums = adj_train.shape[0]
    test_gap = args.test_gap
    nclass = y_train.shape[1]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # set device
    if args.cuda:
        device = torch.device("cuda")
        print("use cuda")
    else:
        device = torch.device("cpu")
    
    # data for train and test
    features = torch.FloatTensor(features).to(device)
    train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]

    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    # init the sampler
    if args.model == 'Fast':
        sampler = Sampler_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)
        model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            sampler=sampler).to(device)
    elif args.model == 'Fast-Alt1':
        sampler = Alternative_Sampler1_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)
        model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            sampler=sampler).to(device)
    elif args.model == 'Fast-Alt2':
        sampler = Alternative_Sampler2_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)
        model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            sampler=sampler).to(device)
    elif args.model == 'AS':
        sampler = Sampler_ASGCN(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                device=device)
        model = GCN(nfeat=features.shape[1],    
            nhid=args.hidden,    
            nclass=nclass,    
            dropout=args.dropout,    
            sampler=sampler).to(device)
    elif args.model == "HybridMethod":
        sampler = Sampler_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)

        model = HybridMethod(nfeat=features.shape[1],    
            nhid=args.hidden,    
            nclass=nclass,    
            dropout=args.dropout,    
            sampler=sampler).to(device)
    else:
        print(f"model name error, no model named {args.model}")
        exit()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss
    # loss_fn = torch.nn.CrossEntropyLoss()
    train_accs = []
    test_accs = []
    train_times = []
    # train and test
    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(np.arange(train_nums),
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap)
        test_loss, test_acc, test_time, test_score = test(test_adj,
                                              test_feats,
                                              test_labels,
                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s, "
              f"test_score: {test_score:.3f}")
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_times.append(train_time)

    #Testing Simple Models
    logistic_model = LogisticRegression()
    mlp = MLPClassifier()
    rf = RandomForestClassifier()
    
    num_classes = 7 if args.dataset == "cora" else 3
    logistic_model.fit(train_features, y_train)
    logistic_acc = logistic_model.score(test_feats[test_index], test_labels)
    logistic_predictions = logistic_model.predict(test_feats[test_index])
    logistic_score = f1_score(torch.from_numpy(logistic_predictions), test_labels, num_classes)
    print ("Logistic Model: ", "Accuracy: ", logistic_acc, "Score: ", logistic_score)
    
    mlp.fit(train_features, y_train)
    mlp_acc = mlp.score(test_feats[test_index], test_labels)
    mlp_predictions = mlp.predict(test_feats[test_index])
    mlp_score = f1_score(torch.from_numpy(mlp_predictions), test_labels, num_classes)
    print ("Mlp Classifier: ", "Accuracy: ", mlp_acc, "Score: ", mlp_score)
    
    rf.fit(train_features, y_train)
    rf_acc = rf.score(test_feats[test_index], test_labels)
    rf_predictions = rf.predict(test_feats[test_index])
    rf_score = f1_score(torch.from_numpy(rf_predictions), test_labels, num_classes)
    print ("Rf Classifier: ", "Accuracy: ", rf_acc, "Score: ", rf_score)
    outp = np.array([train_accs,test_accs,train_times])
    np.savetxt('Fast_Alt_Pubmed.csv',outp,delimiter=',')
