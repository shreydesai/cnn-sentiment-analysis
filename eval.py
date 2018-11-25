import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

from models import TextCNN
from utils import (Vocabulary, load, accuracy, compute_dataset_stats,
                   minibatch_iter)

def get_embedding_type(name):
    return name.rstrip('.th').split('_')[-1]

def import_models(dataset):
    models = {}
    for f in glob.glob('checkpoints/cnn_{}_*'.format(dataset)):
        fname = os.path.split(f)[1]
        embedding_dims = 300
        embedding_type = get_embedding_type(fname)

        X_train, y_train = load('{}_train'.format(dataset))
        vocab = load('{}_vocab'.format(dataset)).vocab

        model = TextCNN(
            dataset=dataset,
            input_size=X_train.shape[1],
            vocab_size=len(vocab) + 1,
            embedding_dims=embedding_dims,
            embedding_type=embedding_type
        )
        model.load_state_dict(torch.load(f))
        model.eval()
        models[fname] = model
    
    return models

def cv_score(dataset, embedding_type, epochs, batch_size=32,
             learning_rate=1e-4, regularization=0):
    kf = KFold(10)
    X, y = load('{}_train'.format(dataset))
    vocab = load('{}_vocab'.format(dataset)).vocab

    cv_acc = []
    cv_std = []

    for ci, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        X_train = torch.as_tensor(X_train, dtype=torch.long).cuda()
        y_train = torch.as_tensor(y_train, dtype=torch.float).cuda()
        X_test = torch.as_tensor(X_test, dtype=torch.long).cuda()
        y_test = torch.as_tensor(y_test, dtype=torch.float).cuda()

        model = TextCNN(
            dataset=dataset,
            input_size=X_train.shape[1],
            vocab_size=len(vocab) + 1,
            embedding_dims=300,
            embedding_type=embedding_type
        ).cuda()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=regularization)
        criterion = nn.BCEWithLogitsLoss()

        model.train()

        for epoch in range(epochs):
            for i, batch in enumerate(minibatch_iter(len(X_train), batch_size)):
                X_train_batch = X_train[batch].cuda()
                y_train_batch = y_train[batch].cuda()
                
                optimizer.zero_grad()

                output = model(X_train_batch)
                train_loss = criterion(output, y_train_batch)

                train_loss.backward()
                optimizer.step()
        
        model.eval()
        _, test_acc, test_std = compute_dataset_stats(
            X_test, y_test, model, nn.BCEWithLogitsLoss(), 256
        )
        
        cv_acc.append(test_acc)
        cv_std.append(test_std)
        print('  [{}] acc={}, std={}'.format(ci+1, test_acc, test_std))
    
    print('{} - {}'.format(dataset, embedding_type))
    print('Mean acc - {}'.format(np.mean(cv_acc)))
    print('Min acc - {}'.format(np.min(cv_acc)))
    print('Max acc - {}'.format(np.max(cv_acc)))
    print('Mean std - {}'.format(np.mean(cv_std)))

def test_score(dataset):
    X_test, y_test = load('{}_test'.format(dataset))
    vocab = load('{}_vocab'.format(dataset)).vocab
    models = import_models(dataset)

    X_test = torch.as_tensor(X_test, dtype=torch.long)
    y_test = torch.as_tensor(y_test, dtype=torch.float)

    for name, model in models.items():
        test_acc = []
        test_std = []
        embedding_type = get_embedding_type(name)

        for i in range(10):
            _, acc, std = compute_dataset_stats(
                X_test.cuda(), y_test.cuda(), model.cuda(),
                nn.BCEWithLogitsLoss(), 256
            )
            test_acc.append(acc)
            test_std.append(std)

        print('{} - {}'.format(dataset, embedding_type))
        print('Mean acc - {}'.format(np.mean(test_acc)))
        print('Min acc - {}'.format(np.min(test_acc)))
        print('Max acc - {}'.format(np.max(test_acc)))
        print('Mean std - {}'.format(np.std(test_std)))

if __name__ == '__main__':
    embedding_types = ['random', 'w2v', 'glove', 'nb']

    print('Evaluating MR dataset')
    mr_epochs = {'random':30,'w2v':25,'glove':23,'nb':13}
    for embedding_type, epochs in mr_epochs.items():
        cv_score('mr', embedding_type, epochs)

    print('Evaluating MPQA dataset')
    mpqa_epochs = {'random':17,'w2v':26,'glove':9,'nb':13}
    for embedding_type, epochs in mpqa_epochs.items():
        cv_score('mpqa', embedding_type, epochs)

    print('Evaluating SST-2 dataset')
    test_score('sst')