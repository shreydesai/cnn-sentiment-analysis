import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import load

class _NgramBlock(nn.Module):

    def __init__(self, input_size, kernel, channels=128):
        super(_NgramBlock, self).__init__()

        self.conv = nn.Conv2d(1, channels, kernel, 1, 0)
        self.pool = nn.MaxPool1d(input_size - kernel[0] + 1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.squeeze(x,-1)
        x = self.pool(x)
        return x.view(-1,128)

class TextCNN(nn.Module):

    def __init__(self, dataset, input_size, vocab_size,
                 embedding_dims, embedding_type, channels=128):
        super(TextCNN, self).__init__()

        if embedding_type == 'random':
            self.embeddings = nn.Embedding(
                vocab_size,
                embedding_dims,
                padding_idx=0
            )
            self.embeddings.weight.requires_grad = True
        else:
            embs_path = '{}_{}_embs'.format(dataset, embedding_type)
            print('Using', embs_path)
            weights = torch.as_tensor(load(embs_path)).float()
            self.embeddings = nn.Embedding.from_pretrained(
                weights,
                freeze=False
            )
        
        self.block1 = _NgramBlock(input_size, (2, embedding_dims))
        self.block2 = _NgramBlock(input_size, (3, embedding_dims))
        self.block3 = _NgramBlock(input_size, (4, embedding_dims))
        self.block4 = _NgramBlock(input_size, (5, embedding_dims))

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(channels*4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU(True)

        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        x = self.embeddings(x)[:,None,:,:]
        x = torch.cat((
            self.block1(x),
            self.block2(x),
            self.block3(x),
            self.block4(x)
        ),-1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x