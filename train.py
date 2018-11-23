import os
import datetime
import argparse
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import TextCNN
from utils import (load, num_batches, minibatch_iter,
                   accuracy, compute_dataset_stats)

class StatsLogger:

    def __init__(self, dirname, stats_dir, name, run_uid):
        stats_path = os.path.join(dirname, stats_dir, '{}_{}.csv'.format(
            name, run_uid
        ))
        self.f = open(stats_path, 'w+')
        self.f.write('epoch,t-loss,t-acc,v-loss,v-acc\n')
    
    def write(self, stats, stdout=True):
        self.f.write((','.join(['{}'] * len(stats)) + '\n').format(*stats))
    
    def close(self):
        self.f.close()

def train(name, dataset, epochs, batch_size, learning_rate, regularization,
          embedding_dims, embedding_type):
    
    dirname, _ = os.path.split(os.path.abspath(__file__))
    run_uid = datetime.datetime.today().strftime('%Y-%m-%dT%H:%M:%S')
    logger = StatsLogger(dirname, 'stats', name, run_uid)

    print('Loading data')
    X_train, y_train = load('{}_train'.format(dataset))
    X_valid, y_valid = load('{}_valid'.format(dataset))
    vocab = load('{}_vocab'.format(dataset))

    X_train = torch.as_tensor(X_train, dtype=torch.long)
    y_train = torch.as_tensor(y_train, dtype=torch.float)
    X_valid = torch.as_tensor(X_valid, dtype=torch.long)
    y_valid = torch.as_tensor(y_valid, dtype=torch.float)

    prev_acc = 0

    model = TextCNN(
        dataset=dataset,
        input_size=X_train.size()[1],
        vocab_size=len(vocab) + 1,
        embedding_dims=embedding_dims,
        embedding_type=embedding_type
    )
    print(model)
    print('Parameters: {}'.format(sum([p.numel() for p in \
                                  model.parameters() if p.requires_grad])))
    print('Training samples: {}'.format(len(X_train)))

    if torch.cuda.is_available():
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_valid = X_valid.cuda()
        y_valid = y_valid.cuda()
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=regularization)
    criterion = nn.BCEWithLogitsLoss()

    print('Starting training')
    for epoch in range(epochs):
        epoch_loss = []
        epoch_acc = []

        iters = 0
        total_iters = num_batches(len(X_train), batch_size)

        for i, batch in enumerate(minibatch_iter(len(X_train), batch_size)):
            model.train()

            X_train_batch = X_train[batch]
            y_train_batch = y_train[batch]

            if torch.cuda.is_available():
                X_train_batch = X_train_batch.cuda()
                y_train_batch = y_train_batch.cuda()
            
            optimizer.zero_grad()

            output = model(X_train_batch)
            train_loss = criterion(output, y_train_batch)
            train_acc = accuracy(output, y_train_batch)

            epoch_loss.append(train_loss.item())
            epoch_acc.append(train_acc.item())

            train_loss.backward()
            optimizer.step()
        
        model.eval()
        train_loss, train_acc = np.mean(epoch_loss), np.mean(epoch_acc)
        valid_loss, valid_acc, _ = compute_dataset_stats(
            X_valid, y_valid, model, nn.BCEWithLogitsLoss(), batch_size*2
        )

        stats = [epoch + 1, train_loss, train_acc, valid_loss, valid_acc]
        epoch_string = '* Epoch {}: t_loss={:.3f}, t_acc={:.3f}, ' + \
                       'v_loss={:.3f}, v_acc={:.3f}'
        print(epoch_string.format(*stats))
        logger.write(stats)
        
        # checkpoint model
        if prev_acc < valid_acc:
            prev_acc = valid_acc
            model_path = os.path.join(dirname, 'checkpoints', '{}_{}_e{}_{}.th'.format(
                name, run_uid, epoch + 1, valid_acc
            ))
            torch.save(model.state_dict(), model_path)
    
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='model name')
    parser.add_argument('-d', '--dataset', help='dataset name')
    parser.add_argument('-e', '--epochs', type=int, help='epochs')
    parser.add_argument('-b', '--batch', type=int, help='batch size')
    parser.add_argument('-l', '--lr', type=float, help='learning rate')
    parser.add_argument('-r', '--reg', type=float, help='regularization')
    parser.add_argument('-m', '--edims', type=int, help='embedding dimensions')
    parser.add_argument('-t', '--etype', help='embedding type')
    args = parser.parse_args()
    print(args)

    train(
        args.name,
        args.dataset,
        args.epochs,
        args.batch,
        args.lr,
        args.reg,
        args.edims,
        args.etype
    )