import re

import numpy as np

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