from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from datetime import datetime
import numpy as np
import argparse
import time
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import precision_recall_curve

from data_loader import NewsEventDataset, DatasetSampler
from gat import GAT

current_time = str(datetime.now())

tweet_id = sys.argv[1]
timestamp_folder = sys.argv[2]

adj_file = tweet_id + '/' + timestamp_folder + '/' + tweet_id + '_adj.npy'
cred_file = tweet_id + '/' + timestamp_folder + '/' + tweet_id + '_cred.npy'
label_file = tweet_id + '/' + timestamp_folder + '/' + tweet_id + '_label.npy'
nbr_file = tweet_id + '/' + timestamp_folder + '/' + tweet_id + '_nbr.npy'
trust_file = tweet_id + '/' + timestamp_folder + '/' + tweet_id + '_trust.npy'


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='L2 Regularization.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--hidden-units', type=str, default="16,8",
                    help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="1,1,1",
                    help="Heads in each layer, splitted with comma")
parser.add_argument('--batch', type=int, default=64, help="Batch size")
parser.add_argument('--check-point', type=int, default=1, help="Check point")
parser.add_argument('--shuffle', action='store_true', default=False, help="Shuffle dataset")
parser.add_argument('--train-ratio', type=float, default=70, help="Training data %")
parser.add_argument('--valid-ratio', type=float, default=15, help="Validation data %")
parser.add_argument('--class-weight-balanced', action='store_true', default=False,
                    help="Adjust weights inversely proportional"
                    " to class frequencies in the input data")
parser.add_argument('--use-vertex-feature', action='store_true', default=False,
                    help="Whether to use vertices' structural features")


args, unknown = parser.parse_known_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = NewsEventDataset(args.seed, args.shuffle, adj_file, cred_file,
                            label_file, nbr_file, trust_file)
N = len(dataset)
n_classes = 2
class_weight = dataset.get_class_weight() \
        if args.class_weight_balanced else torch.ones(n_classes)
feature_dim = dataset.get_credibility_feature_dimension()
n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")] + [n_classes]


train_start,  valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
train_loader = DataLoader(dataset, batch_size=args.batch,
                        sampler=DatasetSampler(valid_start - train_start, 0))
valid_loader = DataLoader(dataset, batch_size=args.batch,
                        sampler=DatasetSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(dataset, batch_size=args.batch,
                        sampler=DatasetSampler(N - test_start, test_start))


n_heads = [int(x) for x in args.heads.strip().split(",")]
model = GAT(vertex_feature=dataset.get_trust_features(),
            use_vertex_feature=args.use_vertex_feature,
            n_units=n_units, n_heads=n_heads,
            dropout=args.dropout)


params = [{'params': model.layer_stack.parameters()}]

optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)


def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    y_score_temp = []
    for i_batch, batch in enumerate(loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        output = model(features, vertices, graph)
        output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        y_score_temp += output.data.tolist()
        total += bs

    model.train()

    if thr is not None:
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    print('Epoch: ', epoch)
    print("{}, loss:{:.3f}, acc:{:.3f}, prec:{:.3f}, rec:{:.3f}, f1:{:.3f}".format(
                log_desc, loss / total, acc,  prec[1], rec[1], f1[1]))

    with open(tweet_id + '/' + timestamp_folder + '/' + log_desc + '_' + current_time + '.txt', "a") as fp:
        fp.write("{},loss:{:.3f},acc:{:.3f},prec:{:.3f},rec:{:.3f},f1:{:.3f}\n".format(
                log_desc, loss / total, acc,  prec[1], rec[1], f1[1]))

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        return best_thr
    else:
        return None


def train(epoch, train_loader, valid_loader, test_loader):

    model.train()

    loss = 0.
    total_loss = 0.
    for i_batch, batch in enumerate(train_loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        optimizer.zero_grad()

        output = model(features, vertices, graph)
        output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total_loss += bs
        loss_train.backward()
        optimizer.step()

    if (epoch + 1) % args.check_point == 0:
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_loader, thr=best_thr, log_desc='test_')


# Training..
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch, train_loader, valid_loader, test_loader)
best_thr = evaluate(args.epochs, valid_loader, return_best_thr=True, log_desc='valid_')


# Testing..
evaluate(args.epochs, test_loader, thr=best_thr, log_desc='test_')