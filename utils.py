import os
import re
import pickle

import numpy as np
import torch

def load(path):
    f = open(os.path.join('datasets', path), 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    return string

def encode_sent(sent, encoding, max_seq_len):
    sent = sent[:max_seq_len]
    sent = [encoding[w] for w in sent]
    sent = np.pad(
        sent,
        (max_seq_len - len(sent), 0),
        'constant',
        constant_values=(0, 0)
    )
    return sent

def num_batches(n, batch_size):
    return int((n - 1) / batch_size) + 1

def minibatch_iter(n, batch_size):
    batches = num_batches(n, batch_size)
    indices = np.random.permutation(n)

    for batch in range(batches):
        start_index = batch * batch_size
        end_index = min((batch + 1) * batch_size, n)
        yield indices[start_index:end_index]

def accuracy(output, labels):
    output = torch.max(output.double(), dim=1)[1]
    labels = torch.max(labels.double(), dim=1)[1]
    return torch.eq(output, labels).double().mean()

def compute_dataset_stats(X, y, model, criterion, batch_size):
    loss_vals = []
    acc_vals = []

    for batch in minibatch_iter(len(X), batch_size):
        X_batch = X[batch]
        y_batch = y[batch]

        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
        
        output = model(X_batch)
        loss = criterion(output, y_batch)
        acc = accuracy(output, y_batch)

        loss_vals.append(loss.item())
        acc_vals.append(acc.item())
    
    return (
        np.mean(loss_vals),
        np.mean(acc_vals),
        np.std(acc_vals)
    )