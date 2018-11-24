import re
import csv
import pickle

import numpy as np
import gensim

from utils import Vocabulary, encode_sent, clean_str

def read_file(path):
    f = open(path)
    lines = [line for line in f.read().split('\n') if len(line) > 0]
    f.close()
    return lines

def save_object(path, obj):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()

def compute_avg_len(X):
    return int(sum(list(map(len, X)))/len(X))

def load_word2vec(path):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    w2v_vocab = {w:i for i,w in enumerate(w2v.vocab)}
    w2v_vectors = [0]*len(w2v.vocab)
    for w,i in w2v_vocab.items():
        w2v_vectors[i] = w2v.get_vector(w)
    return (
        w2v_vocab,
        w2v_vectors
    )

def load_glove(path):
    glove = np.loadtxt(path, dtype='str', comments=None)
    return (
        {w:i for i,w in enumerate(glove[:,0])},
        glove[:,1:].astype('float')
    )

def load_numberbatch(path):
    nb = np.loadtxt(path, dtype='str', comments=None, skiprows=1)
    return (
        {w:i for i,w in enumerate(nb[:,0])},
        nb[:,1:].astype('float')
    )

def create_embeddings(ds_vocab, emb_vocab, emb_vectors, emb_dims):
    found = 0
    embeddings = np.zeros((len(ds_vocab.vocab) + 1, emb_dims))

    for i in range(1, len(ds_vocab.vocab) + 1):
        word = ds_vocab.vocab[i - 1]
        if word in emb_vocab:
            found += 1
            embeddings[i] = emb_vectors[emb_vocab[word]]
        else:
            embeddings[i] = np.random.normal(size=(emb_dims,))
    
    return (embeddings, found)

def create_mr():
    pos = read_file('raw_datasets/rt-polarity.pos')
    neg = read_file('raw_datasets/rt-polarity.neg')

    # build matrices
    X, y = [], []
    for sent in pos:
        X.append(clean_str(sent))
        y.append([0,1])
    for sent in neg:
        X.append(clean_str(sent))
        y.append([1,0])
    
    # build vocab
    mr_vocab = Vocabulary(X)
    print('vocab', len(mr_vocab.vocab))

    # encode sents
    max_seq_len = compute_avg_len(X)
    for i in range(len(X)):
        X[i] = encode_sent(X[i].split(' '), mr_vocab.encoding, max_seq_len)

    # build embeddings
    embeddings = []
    for name, (emb_vocab, emb_vectors) in embeddings_map.items():
        embedding, found = create_embeddings(
            mr_vocab, emb_vocab, emb_vectors, 300
        )
        embeddings.append(embedding)
        print('{} - {}'.format(name, found))
    w2v_embeddings, glove_embeddings, nb_embeddings = embeddings

    # shuffle
    X, y = np.array(X), np.array(y)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split_idx = int(len(X) * 0.9)
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]

    print('train', X_train.shape, y_train.shape)
    print('valid', X_valid.shape, y_valid.shape)

    # save objects
    save_object('datasets/mr_train', (X_train, y_train))
    save_object('datasets/mr_valid', (X_valid, y_valid))
    save_object('datasets/mr_vocab', mr_vocab)
    save_object('datasets/mr_w2v_embs', w2v_embeddings)
    save_object('datasets/mr_glove_embs', glove_embeddings)
    save_object('datasets/mr_nb_embs', nb_embeddings)

def create_mpqa():
    mpqa = read_file('raw_datasets/mpqa.all')

    # build matrices
    X, y = [], []
    for line in mpqa:
        words = line.split(' ')
        label = [0,0]
        label[int(line[0])] = 1
        sent = clean_str(line[1:])
        
        X.append(sent)
        y.append(label)

    # build vocab
    mpqa_vocab = Vocabulary(X)
    print('vocab', len(mpqa_vocab.vocab))

    # encode sents
    max_len = compute_avg_len(X) 
    for i in range(len(X)):
        X[i] = encode_sent(X[i].split(' '), mpqa_vocab.encoding, max_len)
    
    # build embeddings
    embeddings = []
    for name, (emb_vocab, emb_vectors) in embeddings_map.items():
        embedding, found = create_embeddings(
            mpqa_vocab, emb_vocab, emb_vectors, 300
        )
        embeddings.append(embedding)
        print('{} - {}'.format(name, found))
    w2v_embeddings, glove_embeddings, nb_embeddings = embeddings

    # shuffle
    X, y = np.array(X), np.array(y)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split_idx = int(len(X) * 0.9)
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]

    print('train', X_train.shape, y_train.shape)
    print('valid', X_valid.shape, y_valid.shape)

    # save objects
    save_object('datasets/mpqa_train', (X_train, y_train))
    save_object('datasets/mpqa_valid', (X_valid, y_valid))
    save_object('datasets/mpqa_vocab', mpqa_vocab)
    save_object('datasets/mpqa_w2v_embs', w2v_embeddings)
    save_object('datasets/mpqa_glove_embs', glove_embeddings)
    save_object('datasets/mpqa_nb_embs', nb_embeddings)

def create_sst2():
    sst_train = read_file('raw_datasets/stsa.binary.train')
    sst_valid = read_file('raw_datasets/stsa.binary.dev')
    sst_test = read_file('raw_datasets/stsa.binary.test')

    def build_sst_matrices(lines):
        X, y = [], []
        for line in lines:
            words = line.split(' ')
            label = [0,0]
            label[int(line[0])] = 1
            sent = clean_str(line[1:])
            X.append(sent)
            y.append(label)
        return (X, y)

    X_train, y_train = build_sst_matrices(sst_train)
    X_valid, y_valid = build_sst_matrices(sst_valid)
    X_test, y_test = build_sst_matrices(sst_test)

    # build vocab
    master = X_train + X_valid + X_test
    sst_vocab = Vocabulary(master)
    print('vocab', len(sst_vocab.vocab))

    # encode sents
    max_len = compute_avg_len(X_train) 

    def encode_sst_matrices(X):
        for i in range(len(X)):
            X[i] = encode_sent(X[i].split(' '), sst_vocab.encoding, max_len)
        return X
    
    X_train = encode_sst_matrices(X_train)
    X_valid = encode_sst_matrices(X_valid)
    X_test = encode_sst_matrices(X_test)

    # build embeddings
    embeddings = []
    for name, (emb_vocab, emb_vectors) in embeddings_map.items():
        embedding, found = create_embeddings(
            sst_vocab, emb_vocab, emb_vectors, 300
        )
        embeddings.append(embedding)
        print('{} - {}'.format(name, found))
    w2v_embeddings, glove_embeddings, nb_embeddings = embeddings

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_valid, y_valid = np.array(X_valid), np.array(y_valid)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print('train', X_train.shape, y_train.shape)
    print('valid', X_valid.shape, y_valid.shape)
    print('test', X_test.shape, y_test.shape)

    # save objects
    save_object('datasets/sst_train', (X_train, y_train))
    save_object('datasets/sst_valid', (X_valid, y_valid))
    save_object('datasets/sst_test', (X_test, y_test))
    save_object('datasets/sst_vocab', sst_vocab)
    save_object('datasets/sst_w2v_embs', w2v_embeddings)
    save_object('datasets/sst_glove_embs', glove_embeddings)
    save_object('datasets/sst_nb_embs', nb_embeddings)

if __name__ == '__main__':
    # load word2vec embeddings
    print('Loading Word2Vec embeddings')
    w2v_vocab, w2v_vectors = load_word2vec(
        'embeddings/GoogleNews-vectors-negative300.bin'
    )

    # load glove embeddings
    print('Loading GloVe embeddings')
    glove_vocab, glove_vectors = load_glove(
        'embeddings/glove.6B.300d.txt'
    )

    # load numberbatch embeddings
    print('Loading Numberbatch embeddings')
    nb_vocab, nb_vectors = load_numberbatch(
        'embeddings/numberbatch-en-17.06.txt'
    )

    embeddings_map = {
        'w2v': (w2v_vocab, w2v_vectors),
        'glove': (glove_vocab, glove_vectors),
        'nb': (nb_vocab, nb_vectors)
    }

    print('Creating MR dataset')
    create_mr()
    print()

    print('Creating MPQA dataset')
    create_mpqa()
    print()

    print('Creating SST-2 dataset')
    create_sst2()