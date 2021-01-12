import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence

class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, p_dropout,num_class):
        super(Model, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM
        self.enc = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_dropout if num_layers > 1 else 0.)

        self.drop = nn.Dropout(p_dropout)
        if self.enc.bidirectional==False:
            self.classifier=nn.Linear(hidden_dim//2,num_class)
        else:
            self.classifier = nn.Linear(hidden_dim, num_class) #输入batch_size*hidden_dim,输出batch_size*num_class

    def forward(self, x):

        out, _= self.enc(x)
        out = self.drop(out) #[32,80,256]

        # classifier
        logits = self.classifier(out[:, -1, :])
        return logits

